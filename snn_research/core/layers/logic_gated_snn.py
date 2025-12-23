# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (シナプス・スケーリング版)
# 目的: 全結合(100%)を物理的に破壊し、高精度なスパース・コーディングを強制する。

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
        
        # 初期状態: 閾値以下からスタート (最初は接続なし)
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold - 10)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.10 # 目標結合率10%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        mask = self.states > self.threshold
        return mask.to(torch.float32)

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)

        # 電流計算 (加算のみ)
        current = torch.matmul(x, w.t()).view(-1)
        
        # 膜電位更新
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.copy_(v_mem * 0.5 + current) 
        
        # 発火判定 (厳格な閾値)
        spikes = (v_mem >= cast(torch.Tensor, self.adaptive_threshold)).to(torch.float32)
        
        # 修正1: 適格性トレースの更新 (活動の痕跡を薄く長く残す)
        with torch.no_grad():
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """シナプス・スケーリングを伴う適格性学習"""
        with torch.no_grad():
            trace = cast(torch.Tensor, self.eligibility_trace)
            conn_rate = float(self.get_ternary_weights().mean().item())
            
            # 修正2: 報酬の反映 (プラスの時だけ強く定着させ、マイナスの時はトレースを消去)
            if reward > 0:
                self.states.add_(trace * 5.0)
            else:
                self.states.sub_(trace * 2.0)
            
            # 修正3: シナプス・スケーリング (Global Homeostasis)
            # 結合率が目標を超えると、指数関数的に全結合を切断する圧力をかける
            if conn_rate > self.target_conn_rate:
                # 100%飽和を絶対に許さない強力な減衰
                scaling_factor = (conn_rate / self.target_conn_rate) ** 3
                self.states.sub_(0.5 * scaling_factor)
            else:
                # 結合が足りない場合は、わずかな「自発的結合」を促す
                self.states.add_(0.05)

            self.states.clamp_(1, self.max_states)
