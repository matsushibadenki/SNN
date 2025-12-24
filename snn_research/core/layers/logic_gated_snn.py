# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (生存本能・安定化版)
# 目的: 全切断（Conn 0%）を回避し、報酬に基づく有益な接続を保護・育成する。

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
        
        # 初期状態: 接続率 10-15% 程度を維持しやすい分布
        states = torch.randn(out_features, in_features) * 4.0 + (self.threshold - 5.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 6.0))
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
        
        # 接続数に依存しない安定した電流供給
        conn_count = w.sum(dim=1).clamp(min=1.0)
        # スケーリングをsqrt程度に抑え、活動を維持
        current = torch.matmul(x, w.t()).view(-1) / conn_count.sqrt() * 5.0
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.85).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレース: 積極的な蓄積
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 4.0)
            
            # ホメオスタシス: 適度な発火を許容
            self.adaptive_threshold.add_((spikes - 0.1) * 0.1)
            self.adaptive_threshold.clamp_(2.0, 20.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 習熟度に応じた弾性的減衰（生存バイアス）学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.1 else 0.0
            self.proficiency.copy_(self.proficiency * 0.995 + is_success * 0.005)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 学習率のブースト (Conn 0% を防ぐために初期感度を上げる)
            lr = 5.0 * (1.0 - prof * 0.5)
            
            # 弾性的減衰: 接続が少なすぎる時は減衰を止め、多すぎる時だけ強く引く
            conn_rate = float(self.get_ternary_weights().mean().item())
            
            # 基本減衰率の動的調整 (ターゲット 15%)
            base_decay = 0.05 if conn_rate > 0.10 else -0.02 # 10%以下なら逆に底上げ
            self.states.sub_(base_decay)
            
            if modulation > 0:
                # 成功時は強化
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時の弱体化は慎重に
                self.states.sub_(trace * abs(modulation) * lr * 0.1)

            self.states.clamp_(1, self.max_states)
