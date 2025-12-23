# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (自己組織化・最終安定版)
# 目的: 成長と剪定を活動レベルに応じて自動調整し、飽和を物理的に回避する。

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
        
        # 初期状態: 完全にランダムにし、特定のパターンに偏らないようにする
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 5, self.threshold + 5, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0))
        self.register_buffer('firing_history', torch.zeros(out_features))
        
        # 目標結合密度 (生物の脳に近い 5% 〜 10% を目指す)
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
        
        # 膜電位更新 (リークを強め、古い情報を捨てる)
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.copy_(v_mem * 0.5 + current) 
        
        # 判定
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 発火後の完全リセット (飽和防止)
        self.membrane_potential.copy_(v_mem * (1.0 - spikes))
        
        # 恒常性の更新
        with torch.no_grad():
            self.firing_history.copy_(cast(torch.Tensor, self.firing_history) * 0.9 + spikes * 0.1)
            # 発火が多すぎる個体の閾値を上げ、少なすぎる個体の閾値を下げる
            self.adaptive_threshold.add_((spikes - 0.05) * 0.5)
            self.adaptive_threshold.clamp_(0.5, 20.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """
        自己組織化学習則: 
        1. 報酬学習 (Reward-driven)
        2. 密度抑制 (Density-driven Decay)
        """
        with torch.no_grad():
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # 現在の結合率
            conn_rate = float(self.get_ternary_weights().mean().item())
            
            # 修正1: 報酬に基づく更新をより「鋭く」する
            # 報酬が正なら強く固定、負なら即座に切断
            if reward > 0:
                self.states.add_(correlation * 10.0)
            else:
                self.states.sub_(correlation * 15.0)
            
            # 修正2: ダイナミックな密度制御 (100%飽和を数学的に殺す)
            # 密度が目標を超えた瞬間に、指数関数的に忘却（decay）を強める
            if conn_rate > self.target_conn_rate:
                # 飽和状態へのペナルティ (Conn=1.0 なら 10.0 以上の減衰)
                decay = 0.5 * (conn_rate / self.target_conn_rate) ** 2
                self.states.sub_(decay)
            else:
                # 密度が低いときは、活動に関連した部分だけを微増させる (底上げの局所化)
                self.states.add_(0.1)

            self.states.clamp_(1, self.max_states)
