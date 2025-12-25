# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: Adaptive & Normalized)
# 内容: 適応型閾値、重み正規化、カオス的摂動、およびWTAフォールバックを備えたSNNレイヤー

import torch
import torch.nn as nn
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
            self.base_threshold = 1.0 
            trainable = True
            states = torch.randn(out_features, in_features) * std_dev
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
        
        current = torch.matmul(x, w.t())
        v_mem = current 
        
        if self.mode == 'readout':
            threshold = self.adaptive_threshold.unsqueeze(0)
            
            if self.training:
                with torch.no_grad():
                    fire_rate = (v_mem >= threshold).float().mean(dim=0)
                    target_rate = 0.1
                    
                    # 閾値の動的調整 (緩やかに)
                    delta = 0.01 * (fire_rate - target_rate)
                    self.adaptive_threshold.add_(delta)
                    self.adaptive_threshold.clamp_(0.5, 3.0)
        else:
            threshold = self.base_threshold

        spikes = (v_mem >= threshold).float()
        
        # --- Robustness Fix: Winner-Take-All Fallback (Hard Top-K) ---
        # 高ノイズ時に閾値を超えられない場合でも、最も可能性の高いニューロンを発火させる
        if self.mode == 'readout':
            # バッチ内の各サンプルについて、スパイクが発生したか確認
            has_spike = spikes.sum(dim=1) > 0
            
            # スパイクが発生しなかったサンプル（無発火）のみ対象
            if not has_spike.all():
                no_spike_mask = ~has_spike
                # 無発火サンプルのうち、膜電位が最大のニューロンのインデックスを取得
                _, max_indices = v_mem[no_spike_mask].max(dim=1)
                
                # 強制発火 (Top-1 Fallback)
                # maskされた部分に対応するインデックスに1をセット
                spikes[no_spike_mask, max_indices] = 1.0

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
            
            momentum = 0.95
            
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            self.momentum_buffer.mul_(momentum).add_(delta)
            
            self.states.add_(self.momentum_buffer * learning_rate)
            
            # カオス的摂動 (学習停滞防止)
            if random.random() < 0.01: 
                noise = torch.randn_like(self.states) * 0.01 * learning_rate
                self.states.add_(noise)
            
            # 重み正規化 (暴走防止の要)
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features) 
            scale_factor = torch.clamp(target_norm / (norm + 1e-6), max=1.0)
            self.states.mul_(scale_factor)
            
            self.states.clamp_(-20.0, 20.0)
