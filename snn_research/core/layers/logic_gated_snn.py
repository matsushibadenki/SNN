# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (型安全版)
# 目的: 構造的可塑性をシミュレートし、mypyの型チェックをパスする堅牢な実装を提供する。

import torch
import torch.nn as nn
from typing import cast

class LogicGatedSNN(nn.Module):
    """
    Logic-Gated Spiking Neural Network Layer.
    
    特徴:
    1. Dendritic Logic: 入力スパイクに対して論理ゲートを適用。
    2. 1.58-bit Weights: シナプス結合を3値化。
    3. Structural Plasticity: カウントベースの状態遷移による学習。
    """
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.threshold = max_states // 2
        
        # 内部状態バッファ
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 5, self.threshold + 5, (out_features, in_features)
        ).float())
        
        # 膜電位バッファ
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.spike_threshold = 1.0

    @property
    def states(self) -> torch.Tensor:
        """バッファをTensor型として取得"""
        return cast(torch.Tensor, self.synapse_states)

    @property
    def v_mem(self) -> torch.Tensor:
        """膜電位をTensor型として取得"""
        return cast(torch.Tensor, self.membrane_potential)

    def get_ternary_weights(self) -> torch.Tensor:
        """状態に基づき 1.58ビット重みを生成"""
        # Tensor > int の比較結果に対して .float() を適用
        mask = (self.states > self.threshold)
        return mask.to(torch.float32)

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        """順伝播実行"""
        w = self.get_ternary_weights()
        
        if spike_input.dim() == 1:
            spike_input = spike_input.unsqueeze(0)
        
        # 電流計算
        current = torch.matmul(spike_input, w.t())
        
        # 膜電位更新 (inplace加算を回避し型安全に)
        new_v = self.v_mem + current.view(-1)
        
        # 発火判定
        spikes = (new_v >= self.spike_threshold).float()
        
        # リセット処理
        self.membrane_potential.copy_(new_v * (1.0 - spikes))
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> None:
        """局所的な構造的可塑性の更新"""
        with torch.no_grad():
            # プレとポストの相関
            correlation = torch.matmul(post_spikes.unsqueeze(1), pre_spikes.unsqueeze(0))
            
            # 強化
            self.states.add_(correlation)
            
            # 抑圧
            depression_mask = (post_spikes.unsqueeze(1) > 0) & (pre_spikes.unsqueeze(0) == 0)
            self.states[depression_mask] -= 0.5
            
            # 境界制限
            self.states.clamp_(1, self.max_states)

    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, threshold={self.threshold}'
