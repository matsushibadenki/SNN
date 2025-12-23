# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (適格性トレース実装版)
# 目的: 報酬が遅れてやってくる場合でも、直前の成功体験を正確に配線に刻み込む。

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
        
        # 適格性トレース (Eligibility Trace): どの配線が「最近」使われたかを記録
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.trace_decay = 0.8 # トレースの減衰率

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
        v_mem.copy_(v_mem * 0.7 + current) 
        
        # 判定
        spikes = (v_mem >= cast(torch.Tensor, self.adaptive_threshold)).to(torch.float32)
        
        # 修正1: 適格性トレースの更新 (プレとポストが同時に起きた場所をマーク)
        # 行列演算を使わずに、活動があった場所のトレースを増やす
        with torch.no_grad():
            current_trace = torch.outer(spikes, x.view(-1))
            self.eligibility_trace.copy_(cast(torch.Tensor, self.eligibility_trace) * self.trace_decay + current_trace)
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.5)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """適格性トレースに基づいた三因子学習"""
        with torch.no_grad():
            # 修正2: 報酬とトレースの掛け合わせ
            # 直近で「活躍した」配線に対してのみ、報酬（または罰）を与える
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            if reward > 0:
                # 成功した時、その要因となったトレースを永続的な状態へ反映
                self.states.add_(trace * reward * 30.0)
            else:
                # 失敗した時、関与した配線を削る
                self.states.sub_(trace * abs(reward) * 5.0)
            
            # 修正3: 自然なプルーニングと自律的成長のバランス
            conn_rate = float(self.get_ternary_weights().mean().item())
            if conn_rate < 0.05: # 目標密度5%
                self.states.add_(0.1) # 成長
            else:
                self.states.sub_(0.2) # 剪定

            self.states.clamp_(1, self.max_states)
