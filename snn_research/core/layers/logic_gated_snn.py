# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (自己再生・恒常性版)
# 目的: 全消滅を物理的に回避し、情報の「代謝」を回すことで高精度な認識を定着させる。

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
        
        # 初期状態: 閾値付近にバラつかせ、10%程度が接続された状態からスタート
        states = torch.randn(out_features, in_features) * 2.0 + (self.threshold - 2.0)
        mask = torch.rand_like(states) < 0.1
        states[mask] += 5.0
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
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
        
        # 修正1: 入力に微弱な熱ノイズを加え、デッドロックを防ぐ
        current = torch.matmul(x, w.t()).view(-1)
        thermal_noise = torch.randn_like(current) * 0.05
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current + thermal_noise)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレースの記憶時間を少し長く (0.9 -> 0.95)
            self.eligibility_trace.mul_(0.95).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 発火率のホメオスタシス
            self.adaptive_threshold.add_((spikes - 0.05) * 0.1)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 生存本能と情報の代謝を伴う学習則 """
        with torch.no_grad():
            # 習熟度の更新
            self.proficiency.copy_(self.proficiency * 0.99 + (1.0 if reward > 0 else 0.0) * 0.01)
            
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward / 5.0)).item()
            
            # 修正2: 習熟度が低い間は削る力を極めて弱くする (情報の保護)
            pruning_viscosity = torch.clamp(self.proficiency, 0.1, 1.0).item()
            
            if modulation > 0:
                self.states.add_(trace * modulation * 8.0)
            else:
                self.states.sub_(trace * abs(modulation) * 4.0 * pruning_viscosity)
            
            # 修正3: 強力な「自己再生」メカニズム (全消滅の阻止)
            if conn_rate < self.target_conn_rate:
                # 密度が足りないほど、発芽率を上げる
                sprout_prob = (self.target_conn_rate - conn_rate) * 0.1
                revive_mask = torch.rand_like(self.states) < sprout_prob
                self.states[revive_mask] = float(self.threshold + 2.0)
            elif conn_rate > 0.4:
                # 密度過多の場合のみ、自然減衰を強める
                self.states.sub_(0.1)

            self.states.clamp_(1, self.max_states)
