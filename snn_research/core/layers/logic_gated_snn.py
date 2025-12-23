# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (動的密度制御版)
# 目的: 学習のダイナミクスを維持しつつ、最適な結合密度を確保して認識精度を向上させる。

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
        
        # 初期状態
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 5, self.threshold + 5, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0))
        self.register_buffer('firing_history', torch.zeros(out_features))
        
        # 目標結合率 (1.0% は低すぎるため 10.0% 前後を目指す)
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
        
        # 膜電位の更新 (時間的な統合能力を維持)
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.copy_(v_mem * 0.7 + current) 
        
        # 判定
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem + torch.randn_like(v_mem) * 0.1 >= v_th).to(torch.float32)
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.5)
        
        # 履歴更新
        with torch.no_grad():
            self.firing_history.copy_(cast(torch.Tensor, self.firing_history) * 0.99 + spikes * 0.01)
            # 発火頻度が低い場合は閾値を少しずつ下げていく
            self.adaptive_threshold.add_((spikes - 0.02) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """密度制御を伴う三因子学習"""
        with torch.no_grad():
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # 現在の結合率
            conn_rate = float(self.get_ternary_weights().mean().item())
            
            # 修正: 報酬に基づいた更新。LTP(強化)の感度を上げる。
            if reward > 0:
                self.states.add_(correlation * reward * 15.0)
            else:
                self.states.sub_(correlation * abs(reward) * 5.0)
            
            # --- 恒常的密度制御 (Homeostatic Connectivity Control) ---
            # 結合率が目標より低い場合は、ランダムに重みを底上げする
            if conn_rate < self.target_conn_rate:
                # 結合密度が足りないとき、全体的に状態を少し持ち上げる (0.1 〜 0.5 程度)
                self.states.add_(0.2)
            else:
                # 結合密度が過剰なときは忘却を強める
                self.states.sub_(0.2)

            self.states.clamp_(1, self.max_states)
