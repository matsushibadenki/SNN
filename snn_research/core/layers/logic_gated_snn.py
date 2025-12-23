# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (活動強制型・最終安定版)
# 修正内容: 結合率 10% を維持しつつ、発火率が低い場合に閾値を下げて沈黙(Acc 0%)を物理的に破壊する。

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
        
        # 結合率 10% を初期状態で確保
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 2, self.threshold + 2, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0)) # 閾値を下げて開始
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.10 # 目標結合率
        self.target_firing_rate = 0.05 # 目標発火率 (5%)

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
        v_mem.mul_(0.5).add_(current)
        
        # 適応的閾値による発火
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 修正1: 恒常性の更新 (沈黙を破るための強力なフィードバック)
        with torch.no_grad():
            # 発火が目標以下なら閾値を下げ、以上なら上げる
            self.adaptive_threshold.add_((spikes - self.target_firing_rate) * 0.1)
            self.adaptive_threshold.clamp_(0.2, 10.0)
            
            # トレース更新
            self.eligibility_trace.mul_(0.6).add_(torch.outer(spikes, x.view(-1)))
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 密度制御と報酬のバランス """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 報酬による状態遷移 (小刻みに更新)
            modulation = torch.tanh(torch.tensor(reward)).item()
            self.states.add_(trace * modulation * 5.0)
            
            # 修正2: 結合率の安定化門限
            # 10%を基準に、低い時は成長(LTP)、高い時は剪定(LTD)を強制する
            if conn_rate < self.target_conn_rate:
                self.states.add_(0.05)
            else:
                self.states.sub_(0.10)

            self.states.clamp_(1, self.max_states)
