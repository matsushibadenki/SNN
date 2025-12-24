# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: 型安全性向上版)

import torch
import torch.nn as nn
import math
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    # バッファの型ヒントを明示
    membrane_potential: torch.Tensor
    synapse_states: torch.Tensor

    def __init__(self, in_features: int, out_features: int, max_states: int = 100, mode: str = 'reservoir') -> None:
        """
        mode: 
          - 'reservoir': 固定重み、3値量子化 (-1, 0, 1)
          - 'readout': 学習可能、連続値重み (Float)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.mode = mode
        
        # モードに応じた初期化
        if self.mode == 'readout':
            # 読み出し層
            std_dev = 0.05
            self.threshold = 1.0 
            trainable = True
        else:
            # リザーバー層: 入力次元数に応じたスケーリング
            std_dev = 2.0 / math.sqrt(in_features)
            self.threshold = 1.0
            trainable = False
            
        # 重み行列 (States)
        states = torch.randn(out_features, in_features) * std_dev
        self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
        
        # 膜電位モニタリング用バッファ
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        
        self.trainable = trainable

    @property
    def states(self) -> torch.Tensor:
        return self.synapse_states

    def get_effective_weights(self) -> torch.Tensor:
        if self.mode == 'readout':
            return self.states
        else:
            # 3値量子化重み
            w = torch.zeros_like(self.states)
            threshold_val = 0.1 
            w[self.states > threshold_val] = 1.0
            w[self.states < -threshold_val] = -1.0
            return w * 0.5 

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_effective_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 電流計算
        current = torch.matmul(x, w.t())
        
        # 膜電位 (Stateless)
        v_mem = current 
        
        # モニタリング用に保存
        # Mypyエラー回避のため、torch関数を使用し、明示的にTensorとして扱う
        if self.training or not self.training:
            v_mean = torch.mean(v_mem, dim=0).detach()
            self.membrane_potential.copy_(v_mean)

        # 発火判定
        spikes = (v_mem >= 1.0).float()
        
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
            
            lr = 0.05
            
            # Delta Rule
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            self.states.add_(delta * lr)
            self.states.mul_(0.9995) 
            self.states.clamp_(-self.max_states, self.max_states)
