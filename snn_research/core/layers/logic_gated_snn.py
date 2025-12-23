# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (低振動・安定適応版)
# 目的: 0%と100%の極端な振動を抑制し、5-15%の安定したスパース結合を維持する。

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
        
        # 初期状態: 閾値付近に密集させ、微細な変化でON/OFFを切り替えやすくする
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold - 2)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.10 # 目標10%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 電流計算
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 時定数を適度に設定 (0.6)
        v_mem.mul_(0.6).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレース更新
            self.eligibility_trace.mul_(0.7).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 3.0)
        
        # リセット (余剰電位を一部残すことで情報の連続性を保つ)
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.3)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 確率的更新とソフト・リミッターによる安定学習 """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 1. 確率的更新 (一度に全てのシナプスを動かさない)
            update_mask = torch.rand_like(self.states) < 0.2
            
            # 2. 報酬の適用 (クリッピングして安定化)
            modulation = torch.clamp(torch.tensor(reward), -2.0, 2.0).item()
            
            # 3. ソフト・リミッター (目標密度に近づくほど変化を小さくする)
            # 密度が低いときは上昇しやすく、高いときは下降しやすくする
            growth_factor = 1.0 - (conn_rate / (self.target_conn_rate * 2.0))
            
            if modulation > 0:
                # 成功時
                self.states[update_mask] += trace[update_mask] * modulation * 5.0 * growth_factor
            else:
                # 失敗時 (または無報酬時)
                self.states[update_mask] += trace[update_mask] * modulation * 2.0
            
            # 4. 基礎代謝 (飽和を防ぐ弱い力)
            if conn_rate > self.target_conn_rate:
                self.states.sub_(0.1)
            elif conn_rate < 0.02: # 2%を切ったら緊急浮上
                self.states.add_(0.5)

            self.states.clamp_(1, self.max_states)
