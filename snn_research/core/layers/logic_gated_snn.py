# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (拮抗抑制・剪定特化版)
# 目的: Conn: 100% の飽和状態を強制的に破壊し、ターゲットに適応したスパースな結合を形成する。

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
        
        # 初期状態を低めに設定し、最初は結合が少ない状態から始める
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 10, self.threshold + 2, (out_features, in_features)
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
        
        # 1. 膜電位の更新 (減衰を速めて飽和を防ぐ)
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.copy_(v_mem * 0.5 + current) 
        
        # 2. 強力な抑制ノイズ
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 3. 発火後の急峻なリセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        
        # 4. 履歴と閾値の急進的な調整
        with torch.no_grad():
            self.firing_history.copy_(cast(torch.Tensor, self.firing_history) * 0.9 + spikes * 0.1)
            # 発火しすぎている場合は閾値を一気に上げる
            self.adaptive_threshold.add_((spikes - 0.05) * 1.0) 
            self.adaptive_threshold.clamp_(1.0, 50.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """報酬に基づく強力な剪定"""
        with torch.no_grad():
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # 現在の結合率
            conn_rate = self.get_ternary_weights().mean().item()
            
            # 修正1: 飽和(Conn: 100%)に対する指数的な忘却圧力
            saturation_pressure = 2.0 ** (conn_rate * 10.0) if conn_rate > 0.5 else 1.0
            
            # 修正2: 負の報酬(間違い)時の強力な剪定
            if reward < 0:
                # 間違ったパターンの寄与を物理的に焼き切る
                self.states.sub_(correlation * abs(reward) * 50.0)
            else:
                # 正解に近い時のみ、慎重に強化
                self.states.add_(correlation * reward * 2.0)
            
            # 基礎代謝（忘却）: 飽和しているほど強くなる
            self.states.sub_(0.5 * saturation_pressure)
            
            # 最小結合の維持 (全消滅回避)
            if conn_rate < 0.02:
                revive_mask = torch.rand_like(self.states) < 0.05
                self.states[revive_mask] += 10.0

            self.states.clamp_(1, self.max_states)
