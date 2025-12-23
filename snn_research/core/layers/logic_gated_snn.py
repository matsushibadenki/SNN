# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (確率的保護・安定化版)
# 目的: 70%超の精度を10%以下の密度で「安定固定」させ、知能の明滅（振動）を消失させる。

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
        
        # 初期状態: 全結合に近い状態からスタート
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold + 2)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))
        
        self.target_conn_rate = 0.10 # 目標10%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # デジタル累積
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレース更新 (長期的な貢献度を記録)
            self.eligibility_trace.mul_(0.95).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 10.0)
            
            # 発火ホメオスタシス
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 確率的プルーニングによる知能の定着化 """
        with torch.no_grad():
            # 習熟度の平滑化
            is_success = 1.0 if reward > 5.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.9 + is_success * 0.1)
            
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward / 10.0)).item()
            
            # 修正1: 確率的保護。高いトレース（貢献度）を持つ配線は削れにくくする。
            protection = torch.sigmoid(trace - 2.0) 
            
            if modulation > 0:
                # 成功時: 強化
                self.states.add_(trace * modulation * 10.0)
            else:
                # 失敗時: 保護されていない配線を優先的に削る
                self.states.sub_(trace * abs(modulation) * 5.0 * (1.0 - protection))
            
            # 修正2: 習熟度に基づく「ソフト・プルーニング」
            # 目標密度を超えている場合、貢献度の低い配線から順に削る
            if conn_rate > self.target_conn_rate:
                # 削るかどうかの確率判定。習熟度が高いほど、無駄な配線を削る勇気を持つ。
                pruning_pressure = (conn_rate - self.target_conn_rate) * self.proficiency.item()
                decay_mask = (torch.rand_like(self.states) < pruning_pressure * 0.1) & (protection < 0.5)
                self.states[decay_mask] -= 1.0
            elif conn_rate < 0.05:
                # 5%を切った場合は、探索のために一律で浮上
                self.states.add_(0.5)

            self.states.clamp_(1, self.max_states)
