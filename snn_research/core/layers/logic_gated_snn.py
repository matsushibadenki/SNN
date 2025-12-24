# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー
# 目的: 初期の高精度ダイナミクスを復元しつつ、弾性的接続制御によって安定性を確保する。

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
        
        # 初期状態: 閾値周辺にランダム性を待たせ、初期の探索能力を確保
        states = torch.randn(out_features, in_features) * 5.0 + (self.threshold - 5.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 4.0))
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
        
        # 接続数に依存しない安定した電流入力
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 初期の高精度を支えた積分定数 0.8
        v_mem.mul_(0.8).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレース: 積極的な蓄積
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 発火率ホメオスタシス
            self.adaptive_threshold.add_((spikes - 0.1) * 0.1)
            self.adaptive_threshold.clamp_(1.0, 15.0)
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 習熟度とリソース競合を同期させた学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 1.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward / 5.0)).item()
            
            # 積極的な学習ゲイン (初期の成功を逃さない)
            lr = 15.0 * (1.0 - prof * 0.5)
            
            if modulation > 0:
                self.states.add_(trace * modulation * lr)
            else:
                self.states.sub_(trace * abs(modulation) * lr * 0.5)
            
            # 弾性的密度制御: 20%〜40%の範囲を許容し、情報の飽和を防ぐ
            conn_rate = float(self.get_ternary_weights().mean().item())
            if conn_rate > 0.4:
                # 密度超過時は全体的にポテンシャルを下げる
                self.states.sub_(0.5)
            elif conn_rate < 0.2:
                # 密度不足時は底上げ
                self.states.add_(0.2)

            self.states.clamp_(1, self.max_states)
