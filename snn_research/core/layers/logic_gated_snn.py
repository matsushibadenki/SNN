# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (極限スパース・平衡版)
# 目的: Conn: 100% を物理的に破壊し、Acc: 80% を維持しつつ 10% 前後のスパース性を強制実現する。

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
        
        # 初期状態: 完全に疎な状態から開始
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold - 5)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 4.0)) 
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.10 # 目標結合率 10%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 1. デジタル累積 (加算のみ)
        current = torch.matmul(x, w.t()).view(-1)
        
        # 修正1: 側方抑制 (誰かが強く発火したら他を黙らせる)
        if current.max() > 0:
            current = current - (current.mean() * 0.8)
            
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.5).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 修正2: トレースの更新 (活動の因果関係を鋭く記録)
        with torch.no_grad():
            self.eligibility_trace.mul_(0.6).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 2.0)
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.1)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 飽和を許さない「彫刻型」学習則 """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 修正3: 過密ペナルティ (指数関数的に忘却を強める)
            # 結合率が target(10%) を超えると、剪定圧力が劇的に増大する
            pruning_pressure = torch.exp(torch.tensor((conn_rate - self.target_conn_rate) * 20.0)).item()
            pruning_pressure = min(pruning_pressure, 50.0) # 上限設定

            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 修正4: 報酬に基づく更新と剪定の拮抗
            if modulation > 0:
                # 成功時も、密度の圧力があれば強化を抑える
                self.states.add_(trace * modulation * 5.0 / (pruning_pressure + 0.1))
            else:
                # 失敗時はトレース箇所を鋭く剪定
                self.states.sub_(trace * abs(modulation) * 10.0)
            
            # 基礎代謝 (全体的な剪定)
            self.states.sub_(0.2 * pruning_pressure)
            
            # 修正5: 最低限の「発芽」 (全消滅防止)
            if conn_rate < 0.02:
                self.states.add_(1.0)

            self.states.clamp_(1, self.max_states)
