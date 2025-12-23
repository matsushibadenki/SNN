# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (競合再配線・スパース特化版)
# 目的: 全結合飽和(100%)を数学的に破壊し、精度80%を維持したまま10%以下のスパース性を実現する。

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
        
        # 初期状態: 疎な状態(threshold以下)からスタート
        self.register_buffer('synapse_states', torch.randint(1, self.threshold, (out_features, in_features)).float())
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 5.0)) # 閾値を高めに設定
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('accumulated_reward', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 重みを 0/1 に完全分離
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # デジタル累積
        current = torch.matmul(x, w.t()).view(-1)
        
        # 修正1: 側方抑制を内包した膜電位更新 (Winner-Take-All)
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 自分の電位がレイヤーの平均を超えていれば維持、そうでなければ減衰
        v_avg = current.mean()
        v_mem.mul_(0.5).add_(current - v_avg * 0.3)
        
        # 判定
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # トレース更新 (短期活動記憶)
        with torch.no_grad():
            self.eligibility_trace.mul_(0.5).add_(torch.outer(spikes, x.view(-1)))
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes))
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 究極のスパース学習則: 成功に関与しなかった配線を積極的に「捨てる」 """
        with torch.no_grad():
            self.accumulated_reward.copy_(self.accumulated_reward * 0.9 + reward * 0.1)
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            modulation = torch.tanh(self.accumulated_reward).item()
            
            # 修正2: 成功(R>0)なら配線を固定、失敗(R<=0)なら配線を切断
            if modulation > 0:
                # 報酬がある場合、活躍したトレースを大幅に強化
                self.states.add_(trace * modulation * 20.0)
            else:
                # 報酬がない場合、トレースに関わらず全結合を一律に弱体化 (スパース化の圧力)
                self.states.sub_(0.5)

            # 修正3: 構造的可塑性の動的ターゲット
            conn_rate = float(self.get_ternary_weights().mean().item())
            if conn_rate > 0.15: # 15%を超えたら強烈にプルーニング
                self.states.sub_(2.0)
            elif conn_rate < 0.05: # 5%を切ったらランダムに発芽
                revive_mask = torch.rand_like(self.states) < 0.005
                self.states[revive_mask] += 10.0

            self.states.clamp_(1, self.max_states)
