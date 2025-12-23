# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (競合・剪定強化版)
# 目的: 全結合飽和(Conn: 100%)を解消し、誤差に基づく配線剪定を駆動させる。

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
        
        # 初期状態を閾値付近に下げ、飽和を防ぐ
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 5, self.threshold + 5, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.5))
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
        
        # 1. 膜電位の更新
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.add_(current)
        
        # 2. 強力な側方抑制 (Winner-Take-All 的な挙動)
        # 最も電位が高いもの以外を抑制
        if v_mem.max() > 0:
            v_mem.sub_(v_mem.mean() * 0.5)
        
        # 3. 発火判定
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 4. リセット (発火したら完全リセット、そうでなければ少し減衰)
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.7)
        
        # 5. 恒常性 (発火しすぎを抑制)
        with torch.no_grad():
            self.firing_history.copy_(cast(torch.Tensor, self.firing_history) * 0.9 + spikes * 0.1)
            # 発火率が高いほど閾値を急激に上げ、Conn: 100% を阻止
            self.adaptive_threshold.add_((spikes - 0.05) * 0.2)
            self.adaptive_threshold.clamp_(1.0, 20.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """三因子学習則による配線剪定"""
        with torch.no_grad():
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # 修正: 報酬(reward)が負（間違い）のとき、その結合を強力に剪定
            if reward < 0:
                # 間違った発火に寄与した配線を切る
                self.states.sub_(correlation * abs(reward) * 20.0)
            else:
                # 正解に近い場合は結合を強化
                self.states.add_(correlation * reward * 5.0)
            
            # 自然な忘却（スパース性の維持）
            self.states.sub_(0.2)
            self.states.clamp_(1, self.max_states)
