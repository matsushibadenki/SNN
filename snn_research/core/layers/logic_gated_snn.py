# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (アニーリング・バースト版)
# 目的: 安定性を維持しつつ、成功時に一時的な「密度の爆発」を許容し、Acc 50%超の回路を彫り出す。

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
        
        # 閾値より少し下に配置（最初は未結合からスタート）
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold - 5)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.5))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.12 # 安定時のターゲット

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 空間的抑制の緩和（情報の流入を増やす）
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.7).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレース更新（より強く残す）
            self.eligibility_trace.mul_(0.8).add_(torch.outer(spikes, x.view(-1)) * 2.0)
            self.eligibility_trace.clamp_(0, 8.0)
            
            # 発火のホメオスタシス
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 成功時のバーストを許容する動的可塑性 """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 修正1: 成長抑制を「ソフト」に（Conn 30% までは成長を邪魔しない）
            growth_brake = torch.exp(torch.tensor(max(0, conn_rate - 0.3) * 10)).item()
            
            if modulation > 0:
                # 成功時: 制限を気にせず一気に配線（バースト）
                self.states.add_(trace * modulation * 20.0 / growth_brake)
            else:
                # 失敗時: 痕跡箇所を鋭く削除
                self.states.sub_(trace * abs(modulation) * 8.0)
            
            # 修正2: 密度安定化の力を「段階的」に
            if conn_rate > self.target_conn_rate:
                # 目標を超えている時だけ、ゆっくり削る（成功した構造をいきなり壊さない）
                decay_rate = 0.1 * (conn_rate / self.target_conn_rate)
                self.states.sub_(decay_rate)
            elif conn_rate < 0.05:
                # 密度が低すぎる時は、全ニューロンの電位を底上げ（探索の開始）
                self.states.add_(0.5)

            self.states.clamp_(1, self.max_states)
