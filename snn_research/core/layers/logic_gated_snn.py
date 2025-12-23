# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (シナプス固定・安定化版)
# 目的: 成功した配線構造を「長期記憶」として固定し、Accの持続的な上昇を実現する。

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
        
        # 状態の初期化
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 5, self.threshold + 5, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        # ターゲット密度の下限を底上げ
        self.target_conn_rate = 0.12 

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 電流計算
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # トレース更新
        with torch.no_grad():
            self.eligibility_trace.mul_(0.85).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 恒常性: 発火しすぎを抑制
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(1.0, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.3)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 長期増強(LTP)を構造的に固定する学習則 """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 修正1: 報酬のゲインをさらに拡大 (20倍)
            if reward > 0:
                # 成功した経路を強力に「彫り込む」
                self.states.add_(trace * reward * 20.0)
            else:
                # 失敗した経路を「削る」
                self.states.sub_(trace * abs(reward) * 2.0)
            
            # 修正2: 密度に応じた「忘却の適応」
            # 密度が低い時は忘却を止め、高い時(飽和時)だけ削る
            if conn_rate > self.target_conn_rate:
                decay = 0.5 * (conn_rate / self.target_conn_rate)
                self.states.sub_(decay)
            elif conn_rate < 0.05:
                # 5%を切ったら成長を促す
                self.states.add_(0.2)
            
            self.states.clamp_(1, self.max_states)
