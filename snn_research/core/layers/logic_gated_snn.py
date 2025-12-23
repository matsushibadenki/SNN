# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (密度強制適応版)
# 目的: Conn: 100% の飽和を物理的に回避し、情報の差別化を可能にするスパース性を維持する。

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
        
        # 初期状態を閾値以下にし、最初は「接続なし」から始める
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 10, self.threshold - 2, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0))
        self.register_buffer('firing_history', torch.zeros(out_features))
        
        # 目標結合率をさらに厳格化 (5% 〜 15% を維持)
        self.target_conn_rate = 0.10

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
        
        # 膜電位更新
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.copy_(v_mem * 0.6 + current) 
        
        # 適応的閾値
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.4)
        
        # 恒常性の更新
        with torch.no_grad():
            self.firing_history.copy_(cast(torch.Tensor, self.firing_history) * 0.9 + spikes * 0.1)
            # 発火しすぎている場合は閾値を上げ、しなさすぎなら下げる
            self.adaptive_threshold.add_((spikes - 0.05) * 0.5)
            self.adaptive_threshold.clamp_(0.8, 20.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """密度依存型の強力な剪定ルール"""
        with torch.no_grad():
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # 結合率の計算
            current_conn = float(self.get_ternary_weights().mean().item())
            
            # 修正1: 飽和に対する指数的な忘却圧力 (Connが100%に近いほど激しく削る)
            decay_factor = 0.1 * (2.0 ** (current_conn / self.target_conn_rate))
            
            # 修正2: 報酬に基づく更新 (LTPを慎重にし、LTDを鋭くする)
            if reward > 0:
                self.states.add_(correlation * reward * 2.0)
            else:
                self.states.sub_(correlation * abs(reward) * 10.0)
            
            # 恒常的な状態の減衰 (密度制御)
            self.states.sub_(decay_factor)
            
            # 全消滅の回避 (1%以下になった場合のみランダムに再接続)
            if current_conn < 0.01:
                revive_mask = torch.rand_like(self.states) < 0.02
                self.states[revive_mask] += 10.0

            self.states.clamp_(1, self.max_states)
