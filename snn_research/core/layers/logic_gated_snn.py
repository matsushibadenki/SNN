# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (構造的覚醒・適応利得版)
# 目的: 学習初期に爆発的な結合試行を行い、正解パターンを捕獲した後にスパース化へ移行する。

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
        
        # 修正1: 初期状態を閾値の「直上」に設定し、最初は全結合に近い状態から探索を始める
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold + 2)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0)) # 閾値を下げて反応性を高める
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.15 # 最終的な目標密度

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
        v_mem.mul_(0.6).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 修正2: トレースの感度を極限まで上げる
            self.eligibility_trace.mul_(0.7).add_(torch.outer(spikes, x.view(-1)) * 5.0)
            self.eligibility_trace.clamp_(0, 10.0)
            
            # 発火が少なすぎる個体の閾値を急速に下げて「無理やり」発火させる
            self.adaptive_threshold.add_((spikes - 0.1) * 0.1)
            self.adaptive_threshold.clamp_(0.2, 5.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 激しい探索から洗練へ移行する可塑性ルール """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 報酬の反映
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            if modulation > 0:
                # 成功時: 強く固定
                self.states.add_(trace * modulation * 25.0)
            else:
                # 失敗時: 痕跡箇所を大幅に削除
                self.states.sub_(trace * abs(modulation) * 15.0)
            
            # 修正3: 密度の動的平衡 (15% を目指してゆっくり削る)
            if conn_rate > self.target_conn_rate:
                # 密度過多の場合、報酬に関わらず全体を少しずつ削る
                self.states.sub_(0.5)
            elif conn_rate < 0.05:
                # 5% を切った場合は、探索のために一律で浮上させる
                self.states.add_(1.0)

            self.states.clamp_(1, self.max_states)
