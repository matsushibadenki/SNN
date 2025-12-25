# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: Dynamic Relative Threshold)
# 内容: 相対的閾値によるロバストな発火制御、コサイン類似度、正規化された学習則

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    # バッファの型ヒントを明示
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
            # コサイン類似度(x10スケール)用の閾値初期値
            self.base_threshold = 0.5
            trainable = True
            
            # 直交初期化 (維持)
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            # クランプ範囲 [-20, 20]
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            # 適応閾値
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * self.base_threshold)
        else:
            # リザーバー層 (std=3.0)
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
        
        # --- Cosine Similarity Logic ---
        if self.mode == 'readout':
            # 1. 正規化 (L2 Norm)
            x_norm = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)
            w_norm = w / (w.norm(p=2, dim=1, keepdim=True) + 1e-8)
            
            # 2. コサイン類似度
            cosine_sim = torch.matmul(x_norm, w_norm.t())
            
            # 3. スケーリング (10.0に戻す)
            v_mem = cosine_sim * 10.0
        else:
            v_mem = torch.matmul(x, w.t())
        
        # --- Thresholding Logic ---
        if self.mode == 'readout':
            # 適応型閾値 (過去の統計)
            adaptive_th = self.adaptive_threshold.unsqueeze(0)
            
            # 相対的閾値 (現在の信号強度に基づく)
            # 各サンプルの最大膜電位の 80% を最低ラインとする
            # これにより、信号全体が弱くても、ピークに近いニューロンは発火できる
            batch_max_v, _ = v_mem.max(dim=1, keepdim=True)
            relative_th = batch_max_v * 0.8
            
            # 最終的な閾値は、適応閾値と相対閾値の小さい方（またはブレンド）を採用
            # ここでは「発火しやすさ」を優先して、低い方を採用する戦略をとる
            # ただし、ノイズ誤発火を防ぐため、最低限のライン(0.1)は設ける
            effective_threshold = torch.min(adaptive_th, relative_th)
            effective_threshold = effective_threshold.clamp(min=0.1)

            spikes = (v_mem >= effective_threshold).float()
            
            # 学習中の閾値更新
            if self.training:
                with torch.no_grad():
                    fire_rate = spikes.mean(dim=0)
                    target_rate = 0.1
                    delta = 0.01 * (fire_rate - target_rate)
                    self.adaptive_threshold.add_(delta)
                    self.adaptive_threshold.clamp_(0.1, 8.0)
        else:
            spikes = (v_mem >= self.base_threshold).float()
        
        # --- Fallback: Hard Winner-Take-All ---
        # それでも発火しない場合（あるいは競合解決のため）、最強のニューロンのみを発火させる
        if self.mode == 'readout':
            # スパイクが一つもないサンプルに対してTop-1を適用
            has_spike = spikes.sum(dim=1) > 0
            if not has_spike.all():
                no_spike_mask = ~has_spike
                _, max_indices = v_mem[no_spike_mask].max(dim=1)
                spikes[no_spike_mask, max_indices] = 1.0
            
            # 推論時(eval)は、複数発火していても最強の1つに絞る (Sharpening)
            if not self.training:
                # 既にスパイクしているものの中で、v_memが最大のものを残す処理も可能だが、
                # ここではシンプルに argmax を正解とするケースが多いので、
                # 出力スパイクが「確率分布」として扱われるよう、そのままにするか、WTAするか。
                # 精度重視ならWTAを強制する。
                final_spikes = torch.zeros_like(spikes)
                _, max_idx = v_mem.max(dim=1)
                final_spikes.scatter_(1, max_idx.unsqueeze(1), 1.0)
                spikes = final_spikes

        if self.training or not self.training:
            v_mean = torch.mean(v_mem, dim=0).detach()
            self.membrane_potential.copy_(v_mean)

        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor], learning_rate: float = 0.02) -> None:
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # シンプルなデルタ則 (報酬変調)
            momentum = 0.95
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            self.momentum_buffer.mul_(momentum).add_(delta)
            self.states.add_(self.momentum_buffer * learning_rate)
            
            # --- Weight Normalization ---
            # 学習直後に正規化することで、方向ベクトルとしての性質を保つ
            
            # 1. Centering
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # 2. Chaos (微小)
            if random.random() < 0.01: 
                noise = torch.randn_like(self.states) * 0.005 * learning_rate
                self.states.add_(noise)
            
            # 3. Norm Scaling (重要)
            # コサイン類似度のための重みなので、ノルムを一定に保つのが理想的
            # ここでは厳密な1.0ではなく、ある程度の大きさを維持させる
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features) # 分散1.0相当のノルム
            
            # ノルムが小さすぎたり大きすぎたりする場合のみ補正
            scale_factor = target_norm / (norm + 1e-8)
            # 急激な変化を防ぐため、補正係数をクリップする手もあるが、
            # ここでは単純に適用して「球面上」での最適化に近づける
            self.states.mul_(scale_factor)
            
            # 値の暴走を防ぐクランプ
            self.states.clamp_(-20.0, 20.0)
