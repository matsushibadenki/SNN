# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (強制覚醒・定着版)
# 目的: Conn: 0% を物理的に禁止し、情報の「流れ」を常に確保して学習を継続させる。

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
        
        # 初期状態: 20% の配線をランダムに「強く」接続
        states = torch.full((out_features, in_features), float(self.threshold - 10))
        mask = torch.rand_like(states) < 0.20
        states[mask] = float(self.threshold + 10)
        self.register_buffer('synapse_states', states)
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))
        
        self.target_conn_rate = 0.15 # 目標15%

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
        # 修正1: 基底揺らぎの強化 (入力がなくても常にわずかに発火を試みる)
        v_mem.mul_(0.8).add_(current + torch.randn_like(current) * 0.2)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 修正2: トレースの保持時間を極限まで伸ばす (0.95 -> 0.98)
            self.eligibility_trace.mul_(0.98).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 10.0)
            
            # 発火率ホメオスタシス: 沈黙への抵抗を2倍に強化
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.1)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 忘却を抑え、成功を蓄積する適応ルール """
        with torch.no_grad():
            # 習熟度の判定を「わずかな一致」でも許可
            is_growing = 1.0 if reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.995 + is_growing * 0.005)
            
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward / 5.0)).item()
            
            # 修正3: 更新の「非対称性」 (増やす時は速く、削る時は慎重に)
            if modulation > 0:
                self.states.add_(trace * modulation * 5.0)
            else:
                # 習熟度が低い間は、削る力を 1/10 に抑える
                decay_viscosity = max(0.1, self.proficiency.item())
                self.states.sub_(trace * abs(modulation) * 2.0 * decay_viscosity)
            
            # 修正4: 能動的な再配線 (Structural Vitality)
            # 10% を下回ったら強制的に「芽」を吹かせる
            if conn_rate < 0.10:
                revive_prob = (0.10 - conn_rate) * 0.1
                revive_mask = torch.rand_like(self.states) < revive_prob
                self.states[revive_mask] = float(self.threshold + 5.0)
            elif conn_rate > self.target_conn_rate + 0.1:
                # 密度過多の場合のみマイルドに減衰
                self.states.sub_(0.1)

            self.states.clamp_(1, self.max_states)
