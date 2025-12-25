# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: Z-Score Adaptive Gain)
# 内容: Zスコア正規化による動的ゲイン調整、超高ノイズ(0.45)対応、高速化実装

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    # バッファの型ヒント
    membrane_potential: torch.Tensor
    synapse_states: torch.Tensor
    frozen_weight: torch.Tensor
    momentum_buffer: torch.Tensor
    adaptive_threshold: torch.Tensor

    def __init__(self, in_features: int, out_features: int, max_states: int = 100, mode: str = 'reservoir') -> None:
        """
        mode: 
          - 'reservoir': 固定重み、3値量子化。
          - 'readout': 学習可能、連続値重み。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.mode = mode
        
        if self.mode == 'readout':
            # 読み出し層
            std_dev = 0.05
            self.base_threshold = 1.0 # Zスコアベースなので標準偏差の倍率として機能
            trainable = True
            
            # 直交初期化
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            # 適応閾値 (Zスコア閾値として扱う)
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * self.base_threshold)
        else:
            # リザーバー層
            std_dev = 3.0 / math.sqrt(in_features)
            self.base_threshold = 1.0
            trainable = False
            
            if out_features >= in_features:
                w = torch.empty(out_features, in_features)
                nn.init.orthogonal_(w, gain=1.0)
                mask = (torch.rand_like(w) > 0.7).float()
                raw_states = w * mask * (std_dev * 4.0)
            else:
                raw_states = torch.randn(out_features, in_features) * std_dev

            effective_w = self._quantize_weights(raw_states)
            self.register_buffer('frozen_weight', effective_w)
            self.register_buffer('synapse_states', torch.zeros(1))
            self.register_buffer('momentum_buffer', torch.zeros(1))
            self.register_buffer('adaptive_threshold', torch.ones(1))
            
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.trainable = trainable

    @property
    def states(self) -> torch.Tensor:
        return self.synapse_states

    def _quantize_weights(self, x: torch.Tensor) -> torch.Tensor:
        """3値量子化 (-1, 0, 1) * 0.5"""
        w = torch.zeros_like(x)
        threshold_val = 0.01 
        w[x > threshold_val] = 1.0
        w[x < -threshold_val] = -1.0
        return w * 0.5

    def get_effective_weights(self) -> torch.Tensor:
        if self.mode == 'readout':
            return self.states
        else:
            return self.frozen_weight

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_effective_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        if self.mode == 'readout':
            # 1. Cosine Similarity Calculation
            # 正規化して内積をとることで、ベクトルの大きさではなく向きの一致度を見る
            x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            cosine_sim = torch.matmul(x_norm, w_norm.t()) # Shape: [Batch, Out]
            
            # 2. Z-Score Adaptive Gain (The Key Fix)
            # 高ノイズ時はcosine_simの分散が小さくなり、信号が埋もれる。
            # バッチ方向ではなく、"チャネル方向(クラス間)"の統計で正規化することで、
            # 「他のクラスに比べてどれだけ確信度が高いか」を数値化する。
            # これにより、ノイズレベルに関わらず、最もらしい候補が正の大きな値(Z > 2)を持つようになる。
            
            # 平均と標準偏差を計算 (dim=1: クラス方向)
            mean_sim = cosine_sim.mean(dim=1, keepdim=True)
            std_sim = cosine_sim.std(dim=1, keepdim=True)
            
            # Zスコア化 (0除算防止のepsを入れる)
            # 信号が微弱(stdが小)なほど、除算により値が増幅される(Adaptive Gain)
            z_score = (cosine_sim - mean_sim) / (std_sim + 1e-6)
            
            # 3. Softplus & Contrast
            # Zスコアが負（平均以下）のものは抑制し、正のものを非線形に伸ばす
            # Scale x10.0 で扱いやすい膜電位範囲にする
            v_mem = F.softplus(z_score) * 10.0
            
        else:
            # リザーバー層は線形
            v_mem = torch.matmul(x, w.t())
        
        # --- Thresholding Logic ---
        if self.mode == 'readout':
            # 適応型閾値
            adaptive_th = self.adaptive_threshold.unsqueeze(0)
            
            # Zスコアベースなので、相対閾値は不要または固定値でよい。
            # Z=1.5 (上位約7%相当) 以上の興奮があれば発火候補とする。
            # 学習初期は低めに、学習が進むとadaptive_thが効いてくる構成。
            fixed_min_threshold = 10.0 # softplus(1.5)*10 ~= 1.0 * 10 = 10.0
            
            # バッチ内の最大値に対する割合も一応見る（安全策）
            batch_max_v, _ = v_mem.max(dim=1, keepdim=True)
            relative_th = batch_max_v * 0.5
            
            # 最終閾値
            effective_threshold = torch.min(adaptive_th, relative_th)
            # 下限を設定してノイズ発火を抑制
            effective_threshold = effective_threshold.clamp(min=fixed_min_threshold)

            spikes = (v_mem >= effective_threshold).float()
            
            # Homeostasis (閾値調整)
            if self.training:
                with torch.no_grad():
                    fire_rate = spikes.mean(dim=0)
                    target_rate = 0.1 # One-hot target
                    delta = 0.05 * (fire_rate - target_rate)
                    self.adaptive_threshold.add_(delta)
                    # Zスコア空間なので閾値はある程度高くても良い
                    self.adaptive_threshold.clamp_(5.0, 50.0)
        else:
            spikes = (v_mem >= self.base_threshold).float()
        
        # --- Fallback & Sharpening ---
        if self.mode == 'readout':
            # デッドニューロン防止（少なくとも1つ発火）
            has_spike = spikes.sum(dim=1) > 0
            if not has_spike.all():
                no_spike_mask = ~has_spike
                _, max_indices = v_mem[no_spike_mask].max(dim=1)
                spikes[no_spike_mask, max_indices] = 1.0
            
            # 推論時のSharpening (Winner-Take-All)
            if not self.training:
                final_spikes = torch.zeros_like(spikes)
                _, max_idx = v_mem.max(dim=1)
                final_spikes.scatter_(1, max_idx.unsqueeze(1), 1.0)
                spikes = final_spikes

        # 統計記録
        with torch.no_grad():
            v_mean = torch.mean(v_mem, dim=0).detach()
            self.membrane_potential.copy_(v_mean)

        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor], learning_rate: float = 0.02) -> None:
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward_tensor = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward_tensor = reward.unsqueeze(1).expand(-1, self.out_features)
            else:
                reward_tensor = reward
            
            # デルタ則
            delta = torch.matmul(reward_tensor.t(), pre_spikes) / batch_size
            
            # モメンタム
            momentum_factor = 0.95
            self.momentum_buffer.mul_(momentum_factor).add_(delta)
            
            # 重み更新
            self.states.add_(self.momentum_buffer * learning_rate)
            
            # --- Constraints ---
            # 1. Centering (バイアス除去)
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # 2. Norm Scaling (Cos類似度用に正規化)
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features)
            scale_factor = target_norm / (norm + 1e-8)
            self.states.mul_(scale_factor)
            
            self.states.clamp_(-20.0, 20.0)
