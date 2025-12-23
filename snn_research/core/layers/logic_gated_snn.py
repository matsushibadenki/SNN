# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (三因子学習・報酬連動版)
# 修正内容: プレ・ポスト・報酬（誤差）の三因子による学習則を導入し、ターゲットへの収束を導く。

import torch
import torch.nn as nn
from typing import cast

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.threshold = max_states // 2
        
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 5, self.threshold + 15, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0))
        self.register_buffer('firing_history', torch.zeros(out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        mask = self.states > self.threshold
        return mask.to(torch.float32)

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)

        # 電流計算
        current = torch.matmul(x, w.t()).view(-1)
        
        # 膜電位の蓄積とノイズ
        potential = cast(torch.Tensor, self.membrane_potential) + current + torch.randn(self.out_features) * 0.2
        
        # 発火判定
        spikes = (potential >= cast(torch.Tensor, self.adaptive_threshold)).to(torch.float32)
        
        # リセット処理
        self.membrane_potential.copy_(potential * (1.0 - spikes) * 0.5)
        self.firing_history.copy_(cast(torch.Tensor, self.firing_history) * 0.9 + spikes * 0.1)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """
        三因子学習則: ΔW = Reward * (Pre * Post)
        Rewardは誤差の減少度合い、またはターゲットとの一致度。
        """
        with torch.no_grad():
            # 相関 (Eligibility Traceに相当)
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # 修正1: 報酬（reward）に基づいて強化・抑圧を切り替える
            # 正解に近い（reward > 0）なら現在の結合を強く固定
            # 不正解（reward < 0）なら現在の結合を弱体化
            update = correlation * reward * 10.0
            self.states.add_(update)
            
            # 修正2: 恒常的な探索 (ランダムな微増)
            # これがないと一度結合が切れた時に二度と繋がらなくなる
            self.states.add_(torch.randn_like(self.states) * 0.1)

            # 修正3: ターゲットに基づいた直接的な状態誘導 (Supervisor Signal)
            # 全く発火していないのに正解が1の箇所を無理やり Include へ
            # 本来は局所学習に反するが、学習の「種」を蒔くために必要
            dead_mask = (self.firing_history < 0.01)
            self.states[dead_mask] += 0.5

            self.states.clamp_(1, self.max_states)
