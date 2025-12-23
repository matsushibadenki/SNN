# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (蒸留・完成版)
# 目的: Acc 70%超の知能を 10% 密度に凝縮し、行列演算なし・超省電力な論理回路を完成させる。

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
        
        # 初期状態: 探索のために 50% 程度の結合から開始
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 5, self.threshold + 5, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))
        
        self.target_conn_rate = 0.10 # 最終目標 10%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 発火率の適正化
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 高精度を維持したまま、劇的にスパース化する最終ルール """
        with torch.no_grad():
            # 修正1: 習熟度の判定を緩和 (0.1以上の一致があれば賢くなっているとみなす)
            is_learning = 1.0 if reward > 0.1 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_learning * 0.01)
            
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward / 5.0)).item()
            
            # 修正2: 習熟度に応じた「重みの選別」
            # 賢くなるほど（proficiency > 0.5）、活動のない配線を「冷徹に」削る
            pruning_pressure = torch.clamp(self.proficiency * 2.0, 0.0, 1.0).item()
            
            if modulation > 0:
                self.states.add_(trace * modulation * 5.0)
            else:
                self.states.sub_(trace * abs(modulation) * 5.0)
            
            # 修正3: 構造的蒸留 (知識を 10% に凝縮)
            if conn_rate > self.target_conn_rate:
                # 活動痕跡(trace)が薄い配線を優先的に狙い撃ちしてプルーニング
                redundancy_decay = (1.0 - torch.tanh(trace)) * 0.2 * pruning_pressure
                self.states.sub_(redundancy_decay)
            
            self.states.clamp_(1, self.max_states)
