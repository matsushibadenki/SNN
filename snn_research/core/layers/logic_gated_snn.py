# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (適応的成長版)
# 目的: 結合密度を 5-10% まで意図的に引き上げ、認識に必要な情報伝達容量を確保する。

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
        
        # 初期状態: 閾値のすぐ下に設定し、学習による「浮上」を待つ
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold - 5)))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0))
        self.register_buffer('firing_history', torch.zeros(out_features))
        
        # 目標結合率を確実に維持する設定
        self.target_conn_rate = 0.08 

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
        
        # 膜電位の更新
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.copy_(v_mem * 0.8 + current) 
        
        # 判定 (ノイズを減らし、決定論的な寄与を強める)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.5)
        
        # 履歴更新
        with torch.no_grad():
            self.firing_history.copy_(cast(torch.Tensor, self.firing_history) * 0.9 + spikes * 0.1)
            # 発火の平準化
            self.adaptive_threshold.add_((spikes - 0.05) * 0.2)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """成長と剪定のダイナミックな均衡"""
        with torch.no_grad():
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # 現在の結合率
            current_conn = float(self.get_ternary_weights().mean().item())
            
            # 1. 報酬学習則
            if reward > 0:
                # 正解時: 結合を強力に強化し、閾値を突破させる
                self.states.add_(correlation * 20.0)
            else:
                # 不正解時: 関係した配線をマイルドに削る
                self.states.sub_(correlation * 5.0)
            
            # 2. 恒常的成長圧力 (Structural Growth)
            # 目標密度に達するまで、全シナプスを「底上げ」する
            if current_conn < self.target_conn_rate:
                # 結合が足りない時、ランダムではなく全体を押し上げて「候補」を作る
                self.states.add_(0.5)
            else:
                # 密度を超えたら、活動していない結合から順に削る
                self.states.sub_(0.2)

            self.states.clamp_(1, self.max_states)
