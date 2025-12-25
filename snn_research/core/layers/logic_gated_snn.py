# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: High-Contrast & Optimized)
# 内容: 最適化されたコサイン類似度、強化されたコントラスト(x20), ロバストな相対閾値(0.6)

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
            # コサイン類似度(x20スケール)用の閾値初期値
            self.base_threshold = 0.5
            trainable = True
            
            # 直交初期化
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
        
        # --- Optimized Cosine Similarity Logic ---
        if self.mode == 'readout':
            # F.normalize を使用して高速化かつ安定化 (L2 Norm)
            # epsを含めることでゼロ除算を防ぐ
            x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            
            # コサイン類似度計算
            cosine_sim = torch.matmul(x_norm, w_norm.t())
            
            # スケーリング (x14.0 -> x20.0 に変更してコントラストを強化)
            # 高ノイズ下での分離能力を高める
            v_mem = cosine_sim * 20.0
        else:
            v_mem = torch.matmul(x, w.t())
        
        # --- Thresholding Logic ---
        if self.mode == 'readout':
            # 適応型閾値
            adaptive_th = self.adaptive_threshold.unsqueeze(0)
            
            # 相対的閾値 (Batch-wise Adaptive)
            # 0.7 -> 0.6 に緩和。
            # ノイズが高い場合、最大応答も低下するため、少し余裕を持たせて発火させる。
            # 誤発火は後段のSoftmaxやWinner-Take-Allで抑制可能。
            batch_max_v, _ = v_mem.max(dim=1, keepdim=True)
            relative_th = batch_max_v * 0.6
            
            # 最終的な閾値の決定
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
                    # スケール20.0に合わせて上限も少し調整
                    self.adaptive_threshold.clamp_(0.1, 15.0)
        else:
            spikes = (v_mem >= self.base_threshold).float()
        
        # --- Fallback: Hard Winner-Take-All ---
        # 少なくとも1つは発火させる（デッドニューロン防止）
        if self.mode == 'readout':
            has_spike = spikes.sum(dim=1) > 0
            if not has_spike.all():
                no_spike_mask = ~has_spike
                _, max_indices = v_mem[no_spike_mask].max(dim=1)
                spikes[no_spike_mask, max_indices] = 1.0
            
            # 推論時(eval)は Sharpening を適用して明確な予測を行う
            if not self.training:
                final_spikes = torch.zeros_like(spikes)
                _, max_idx = v_mem.max(dim=1)
                final_spikes.scatter_(1, max_idx.unsqueeze(1), 1.0)
                spikes = final_spikes

        # 膜電位の統計記録
        if self.training or not self.training:
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
                # floatの場合はスカラー乗算で済むため拡張しないほうがメモリ効率が良いが
                # 実装の統一性を優先してテンソル計算を行う
                reward_tensor = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward_tensor = reward.unsqueeze(1).expand(-1, self.out_features)
            else:
                reward_tensor = reward
            
            # シンプルなデルタ則 (Reward-Modulated Hebbian Like)
            momentum = 0.95
            
            # delta = (Reward^T @ Pre) / Batch
            delta = torch.matmul(reward_tensor.t(), pre_spikes) / batch_size
            
            self.momentum_buffer.mul_(momentum).add_(delta)
            self.states.add_(self.momentum_buffer * learning_rate)
            
            # --- Weight Normalization & Constraints ---
            
            # 1. Centering (平均を0に保つことでドリフトを防止)
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # 2. Chaos Injection
            # 収束期に入っているため頻度を下げているが、
            # 学習率が小さい場合はノイズもスケーリングして小さくする
            if random.random() < 0.001: 
                noise = torch.randn_like(self.states) * 0.005 * learning_rate
                self.states.add_(noise)
            
            # 3. Norm Scaling (Cosine Simのために球面上に射影)
            # F.normalizeは新しいTensorを返すため、ここではin-placeで計算
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features) # 期待されるノルム長
            # ゼロ除算回避
            scale_factor = target_norm / (norm + 1e-8)
            self.states.mul_(scale_factor)
            
            # Clamp (極端な値の抑制)
            self.states.clamp_(-20.0, 20.0)
