# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (非線形樹状突起版)
# 目的: 単一ニューロン内で非線形な特徴統合を行い、認識精度（Acc）をさらに引き上げる。

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
        
        # 状態の初期化: 1.58-bit (Include/Exclude)
        self.register_buffer('synapse_states', torch.randn(out_features, in_features) * 2.0 + self.threshold)
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('accumulated_reward', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 修正1: 樹状突起による非線形統合 (Dendritic Nonlinearity)
        # 単純な加算の前に、入力と重みの「一致度」を非線形に強調する
        raw_current = torch.matmul(x, w.t()).view(-1)
        
        # 局所的なスパイクが集中した場合、電位を二乗で加速（ボトムアップ・アテンション）
        dendritic_boost = torch.pow(raw_current / (raw_current.mean() + 1.0), 2)
        current = raw_current + dendritic_boost * 0.5
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.7).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレース更新 (因果関係の保持)
            self.eligibility_trace.mul_(0.8).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        with torch.no_grad():
            self.accumulated_reward.copy_(self.accumulated_reward * 0.95 + reward * 0.05)
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 修正2: 報酬の非線形ゲイン
            # 良い結果（正の報酬）に対して、より敏感に反応する
            modulation = torch.tanh(self.accumulated_reward).item()
            gain = 10.0 if modulation > 0 else 5.0
            
            self.states.add_(trace * modulation * gain)
            
            # 修正3: 構造的可塑性の動的調整 (ターゲット密度 10% を維持)
            conn_rate = float(self.get_ternary_weights().mean().item())
            if conn_rate > 0.10:
                self.states.sub_(0.15)
            elif conn_rate < 0.05:
                self.states.add_(0.10)
            else:
                # 安定領域では探索のために少しだけ揺らす
                self.states.add_(torch.randn_like(self.states) * 0.05)

            self.states.clamp_(1, self.max_states)
