# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (自動探索・生命維持版)
# 目的: Conn: 0% を物理的に不可能にし、未探索の入力ビットを優先的に接続して知能を再点火する。

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
        
        # 修正1: 初期状態を閾値付近にバラつかせ、多様性を確保
        self.register_buffer('synapse_states', torch.randn(out_features, in_features) * 2.0 + self.threshold - 1.0)
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.5))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.20 # 目標結合率 20%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 電流計算 (加算のみ)
        current = torch.matmul(x, w.t()).view(-1)
        
        # 修正2: 膜電位の「基底活動」 (入力がなくても常にわずかに揺らす)
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current + torch.randn_like(current) * 0.1)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレース更新
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # ホメオスタシス: 発火がなければ閾値を下げ続ける
            self.adaptive_threshold.add_((spikes - 0.05) * 0.05)
            self.adaptive_threshold.clamp_(0.3, 5.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.5)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 能動的な再配線(Rewiring)アルゴリズム """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 修正3: 報酬学習の安定化
            self.states.add_(trace * modulation * 5.0)
            
            # 修正4: 能動的再配線 (Structural Plasticity)
            # 目標密度(20%)を維持するための強力なフィードバック
            if conn_rate < self.target_conn_rate:
                # 密度が足りない時、ランダムに「芽」を吹かせる
                sprout_prob = (self.target_conn_rate - conn_rate) * 0.2
                revive_mask = torch.rand_like(self.states) < sprout_prob
                self.states[revive_mask] += 10.0 # 閾値を超えるエネルギー
            else:
                # 密度過多の場合のみ剪定
                self.states.sub_(0.1)

            self.states.clamp_(1, self.max_states)
