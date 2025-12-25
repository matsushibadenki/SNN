# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: Scale-up & Fine-tuning)

import torch
import torch.nn as nn
import math
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    # バッファの型ヒントを明示
    membrane_potential: torch.Tensor
    synapse_states: torch.Tensor
    frozen_weight: torch.Tensor
    momentum_buffer: torch.Tensor

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
            self.threshold = 1.0 
            trainable = True
            states = torch.randn(out_features, in_features) * std_dev
            self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
        else:
            # リザーバー層
            # 【修正】1.0 -> 1.2: ニューロン数を増やすため、個々の発火率を少し上げて情報伝達量を確保する
            std_dev = 1.2 / math.sqrt(in_features)
            self.threshold = 1.0
            trainable = False
            
            raw_states = torch.randn(out_features, in_features) * std_dev
            effective_w = self._quantize_weights(raw_states)
            self.register_buffer('frozen_weight', effective_w)
            
            self.register_buffer('synapse_states', torch.zeros(1))
            self.register_buffer('momentum_buffer', torch.zeros(1))
            
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.trainable = trainable

    @property
    def states(self) -> torch.Tensor:
        return self.synapse_states

    def _quantize_weights(self, x: torch.Tensor) -> torch.Tensor:
        """3値量子化 (-1, 0, 1) * 0.5"""
        w = torch.zeros_like(x)
        threshold_val = 0.1 
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
        
        if self.training or not self.training:
            v_mean = torch.mean(v_mem, dim=0).detach()
            self.membrane_potential.copy_(v_mean)

        spikes = (v_mem >= self.threshold).float()
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # ハイパーパラメータ
            # 【修正】0.025 -> 0.02: より慎重に、細かく学習する
            lr = 0.02
            momentum = 0.9
            
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            self.momentum_buffer.mul_(momentum).add_(delta)
            self.states.add_(self.momentum_buffer * lr)
            
            # 正則化
            # 【修正】微弱なWeight Decay (1e-6) を復活させ、極端な重みの偏りを防ぐ
            self.states.mul_(0.999999) 
            self.states.clamp_(-self.max_states, self.max_states)
