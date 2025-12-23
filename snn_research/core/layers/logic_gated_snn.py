# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (再生・安定版)
# 目的: Conn: 0.0% の全消滅を回避し、疎な結合を維持しながら学習を継続させる。

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
        
        # 初期状態: 閾値付近でランダムに
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 2, self.threshold + 8, (out_features, in_features)
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
        
        # 1. 膜電位の更新 (リーキーな特性を強める)
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.copy_(v_mem * 0.9 + current) # 積分特性
        
        # 2. 確率的要素
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem + torch.randn_like(v_mem) * 0.1 >= v_th).to(torch.float32)
        
        # 3. リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.5)
        
        # 4. 履歴と恒常性
        with torch.no_grad():
            self.firing_history.copy_(cast(torch.Tensor, self.firing_history) * 0.99 + spikes * 0.01)
            # 発火しなさすぎを検知して閾値を下げる
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 5.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """全消滅を防ぐ適応的学習則"""
        with torch.no_grad():
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # 修正1: 報酬による更新の安定化 (極端な値を避ける)
            # reward の影響を tanh で制限
            clamped_reward = torch.tanh(torch.tensor(reward))
            
            # 強化と剪定のバランスを調整
            if clamped_reward >= 0:
                self.states.add_(correlation * clamped_reward * 2.0)
            else:
                self.states.sub_(correlation * abs(clamped_reward) * 1.0)
            
            # 修正2: 恒常的な微弱な忘却 (過剰結合の防止)
            self.states.sub_(0.05)
            
            # 修正3: 全消滅回避ロジック (Min Connectivity)
            # 結合率が 5% を切ったらランダムに結合を復活させる
            conn_mask = self.states > self.threshold
            if conn_mask.float().mean() < 0.05:
                # 死んでいるシナプスの中からランダムに選び、Include 状態へ戻す
                revive_mask = (torch.rand_like(self.states) < 0.01) & (~conn_mask)
                self.states[revive_mask] = float(self.threshold + 5)

            self.states.clamp_(1, self.max_states)
