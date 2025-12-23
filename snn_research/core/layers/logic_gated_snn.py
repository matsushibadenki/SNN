# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (再始動・構造進化版)
# 目的: 全結合による沈黙を物理的に破壊し、Acc 立ち上がりを確認した後にスパース化を行う。

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
        
        # 修正1: 最初は「まばら(5%)」な接続から開始し、全結合の死を回避
        initial_states = torch.full((out_features, in_features), float(self.threshold - 5))
        mask = torch.rand_like(initial_states) < 0.05
        initial_states[mask] = float(self.threshold + 5)
        self.register_buffer('synapse_states', initial_states)
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0)) # 閾値を極限まで下げて開始
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))
        
        self.target_conn_rate = 0.10

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        current = torch.matmul(x, w.t()).view(-1)
        
        # 修正2: 沈黙打破 (全員が沈黙している場合、ランダムな1人を強制発火させる準備)
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 発火が全くなければ、閾値を急速に下げる
            if spikes.sum() == 0:
                self.adaptive_threshold.sub_(0.2)
            else:
                self.adaptive_threshold.add_((spikes - 0.1) * 0.1)
            self.adaptive_threshold.clamp_(0.1, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        with torch.no_grad():
            is_success = 1.0 if reward > 5.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward / 10.0)).item()
            
            # 修正3: 積極的な結合生成 (学習初期)
            if self.proficiency < 0.2:
                # まだ下手なうちは、報酬に関わらず「当たったかも」しれない配線を増やす
                if modulation >= 0:
                    self.states.add_(trace * 5.0)
            else:
                # 賢くなってきたら、厳格に強化と削除を行う
                if modulation > 0:
                    self.states.add_(trace * modulation * 10.0)
                else:
                    self.states.sub_(trace * abs(modulation) * 5.0)
            
            # 修正4: 密度調整 (目標10%に向けて常に弱い圧力をかける)
            if conn_rate > self.target_conn_rate:
                self.states.sub_(0.1)
            elif conn_rate < 0.02:
                # 死滅防止
                revive_mask = torch.rand_like(self.states) < 0.01
                self.states[revive_mask] += 10.0

            self.states.clamp_(1, self.max_states)
