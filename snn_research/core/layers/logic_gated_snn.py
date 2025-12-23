# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (自発発火・再点火版)
# 目的: Conn: 0% による冬眠状態を物理的に不可能にし、常に情報を探索させる。

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
        
        # 初期状態: 閾値付近にバラつかせ、一部を接続
        self.register_buffer('synapse_states', torch.randn(out_features, in_features) * 5.0 + self.threshold)
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.08 # 目標結合率 8%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 1. 電流計算 + 熱ゆらぎ(ノイズ)
        current = torch.matmul(x, w.t()).view(-1)
        thermal_noise = torch.randn_like(current) * 0.5 
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current + thermal_noise)
        
        # 2. 活動強制 (沈黙が続く場合は自動で電位を底上げ)
        if v_mem.max() < 0.1:
            v_mem.add_(0.5)
            
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 3. 適格性トレースの更新
        with torch.no_grad():
            self.eligibility_trace.mul_(0.8).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.3)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 生存本能（確率的再結合）を伴う学習則 """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 報酬による変化 (安定性を考え tanh を使用)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 報酬が正なら強化、負なら抑制
            self.states.add_(trace * modulation * 5.0)
            
            # 修正4: 自律的再結合 (Conn 0% 回避)
            # 目標密度を下回っている場合、ランダムに配線を「発芽」させる
            if conn_rate < self.target_conn_rate:
                revive_prob = (self.target_conn_rate - conn_rate) * 0.1
                revive_mask = torch.rand_like(self.states) < revive_prob
                self.states[revive_mask] = float(self.threshold + 5)
            else:
                # 密度過多の場合は基礎代謝で削る
                self.states.sub_(0.1)

            self.states.clamp_(1, self.max_states)
