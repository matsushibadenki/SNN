# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 高精度・動的密度制御版ロジックゲートレイヤー
# 目的: 学習進捗に応じた接続密度制御を導入し、認識精度を最大化する。

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
        
        # 初期状態: 接続率50%付近からスタート
        states = torch.randn(out_features, in_features) * 2.0 + self.threshold
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 5.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))

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
        # 情報を拾いやすくするため減衰を0.9に緩和
        v_mem.mul_(0.9).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレースの蓄積を強化
            self.eligibility_trace.mul_(0.8).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 発火ターゲット 10%
            self.adaptive_threshold.add_((spikes - 0.1) * 0.1)
            self.adaptive_threshold.clamp_(1.0, 20.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.1)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        with torch.no_grad():
            is_success = 1.0 if reward > 0.5 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = self.proficiency.item()
            
            # 学習率の動的制御
            lr = 10.0 * (1.0 - prof * 0.5)
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.clamp(torch.tensor(reward), -1.0, 1.0).item()
            
            if modulation > 0:
                # 成功時: 強く固定
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時: 寄与した接続をリセット
                self.states.sub_(trace * abs(modulation) * lr * 0.5)
            
            # 接続密度の動的ターゲット
            # 初回は30%、習熟したら15%へ絞り込む
            target_conn = 0.3 - (prof * 0.15)
            conn_rate = float(self.get_ternary_weights().mean().item())
            
            # 密度をターゲットに近づける力
            adj_force = 0.5 if conn_rate < target_conn else -0.5
            self.states.add_(adj_force)

            self.states.clamp_(1, self.max_states)
