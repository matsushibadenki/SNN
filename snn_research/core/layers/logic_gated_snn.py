# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (知能結晶化・スパース完成版)
# 目的: Acc 80%超の知能を維持したまま、結合密度を 10-20% まで削ぎ落とし、真の効率を実現する。

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
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold + 5)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.5))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.15 # 目標スパース密度 15%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        current = torch.matmul(x, w.t()).view(-1)
        
        # 側方抑制を適正化
        if current.max() > 0:
            current = current - (current.mean() * 0.5)
            
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレース更新
            self.eligibility_trace.mul_(0.8).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 発火ホメオスタシス
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 知能を維持したまま贅肉を削ぎ落とす(LTD強化型) """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            if modulation > 0:
                # 成功時: 既存の「正解配線」をより深く固定し、それ以外を増やさない
                growth_inhibitor = max(0.1, 1.0 - (conn_rate / self.target_conn_rate))
                self.states.add_(trace * modulation * 5.0 * growth_inhibitor)
            else:
                # 失敗時: 痕跡箇所を鋭く削除
                self.states.sub_(trace * abs(modulation) * 10.0)
            
            # 修正: 強力な「知能維持型プルーニング」
            # 精度が高い（Rewardが良い）時ほど、活動していないシナプスを積極的に削る
            if modulation > 0.5:
                # 成功しているなら、活動痕跡のない(traceが低い)結合は「不要な贅肉」とみなす
                redundancy_mask = (trace < 0.1) & (self.states > self.threshold)
                self.states[redundancy_mask] -= 1.0
            
            # 密度安定化
            if conn_rate > self.target_conn_rate:
                self.states.sub_(0.2 * (conn_rate / self.target_conn_rate))
            elif conn_rate < 0.05:
                self.states.add_(0.5)

            self.states.clamp_(1, self.max_states)
