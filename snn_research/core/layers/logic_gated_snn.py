# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (多重時間スケール学習版)
# 目的: 長期記憶と短期記憶を分離し、誤差の確実な減少と安定したスパース構造を両立させる。

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
        self.register_buffer('synapse_states', torch.randn(out_features, in_features) * 2.0 + self.threshold)
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.5))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        # 修正1: 報酬の移動平均（感情バッファ）
        self.register_buffer('accumulated_reward', torch.zeros(1))
        
        # 修正2: 個別ニューロンの減衰時定数 (多様性による同期回避)
        self.register_buffer('decay_constants', torch.linspace(0.6, 0.9, out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 重みの1.58ビット化（0/1）
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        current = torch.matmul(x, w.t()).view(-1)
        
        # 修正3: 個別の時定数を用いた膜電位更新
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(self.decay_constants).add_(current)
        
        # 判定 (決定論的だが個別の閾値を持つ)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 適格性トレース (短期的な活動記録)
        with torch.no_grad():
            self.eligibility_trace.mul_(0.85).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 4.0)
        
        # リセット処理
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.3)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """多重時間スケールによる学習則"""
        with torch.no_grad():
            # 報酬の積分 (短期的な変動をフィルタリング)
            self.accumulated_reward.copy_(self.accumulated_reward * 0.9 + reward * 0.1)
            
            # 適格性トレース
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 修正4: 累積報酬（安定した信号）に基づく長期記憶の更新
            # 正の累積報酬があるときだけ、現在のトレースを強く固定する
            modulation = torch.tanh(self.accumulated_reward).item()
            
            if modulation > 0:
                self.states.add_(trace * modulation * 5.0)
            else:
                self.states.sub_(trace * abs(modulation) * 2.0)
            
            # 修正5: 競争的可塑性 (Global Competition)
            # 他のニューロンが強く発火した場所を自動的に弱める（側方抑制の重み版）
            conn_rate = float(self.get_ternary_weights().mean().item())
            
            # 密度に応じた自動バランス
            if conn_rate > 0.12:
                self.states.sub_(0.15)
            elif conn_rate < 0.06:
                self.states.add_(0.10)
            else:
                self.states.sub_(0.01) # 基礎代謝

            self.states.clamp_(1, self.max_states)
