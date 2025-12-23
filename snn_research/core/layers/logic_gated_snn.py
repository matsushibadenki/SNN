# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (高密度・高利得学習版)
# 修正内容: 結合密度を10%以上に引き上げ、成功報酬を非線形に増幅することで認識精度を飛躍させる。

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
        
        # 初期状態: 閾値付近に設定し、学習の初動を速める
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 5, self.threshold + 5, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.15 # 目標結合率を15%に設定
        self.target_firing_rate = 0.10 # 目標発火率10%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        # 修正1: 入力強度のブースト (情報の解像度を上げる)
        x = (spike_input * 2.0).clamp(0, 1) if spike_input.dim() > 1 else (spike_input * 2.0).clamp(0, 1).unsqueeze(0)
        
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.7).add_(current)
        
        # 適応的閾値
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 修正2: 恒常性の感度を調整 (沈黙をより強く嫌う)
            self.adaptive_threshold.add_((spikes - self.target_firing_rate) * 0.05)
            self.adaptive_threshold.clamp_(0.5, 5.0)
            
            # トレース更新
            self.eligibility_trace.mul_(0.8).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 3.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.3)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 修正3: 報酬の非線形増幅 (成功への強烈なインセンティブ)
            if reward > 0:
                # 報酬が正なら、その寄与を指数関数的に焼き付ける
                self.states.add_(trace * reward * 15.0)
            else:
                # 報酬が負なら、マイルドに剪定
                self.states.sub_(trace * abs(reward) * 2.0)
            
            # 修正4: 目標密度への強力な誘引 (15%付近で安定させる)
            growth_pressure = (self.target_conn_rate - conn_rate) * 2.0
            self.states.add_(growth_pressure)

            self.states.clamp_(1, self.max_states)
