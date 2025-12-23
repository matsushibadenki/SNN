# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (局所競争・非線形覚醒版)
# 修正内容: 安定した結合率を維持しつつ、報酬に応じた非線形な「配線固定」を導入してAccを向上させる。

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
        
        # 状態の初期化: 閾値のすぐ下に設定
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold - 1)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.5))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        self.target_conn_rate = 0.08 # 目標を8%に微調整

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 重みを 0/1 に完全分離
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 修正1: 空間的抑制 (Spatial Inhibition) 
        # 入力スパイクが多すぎる場合、信号を抑制してスパース性を保つ
        current = torch.matmul(x, w.t()).view(-1)
        if current.sum() > 0:
            current = current / (current.mean() + 1.0) 
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.6).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 修正2: トレースの非線形更新
        # プレ・ポストが一致した際、痕跡をより「深く」残す
        with torch.no_grad():
            self.eligibility_trace.mul_(0.5).add_(torch.outer(spikes, x.view(-1)) * 2.0)
            self.eligibility_trace.clamp_(0, 5.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.2)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 報酬に基づく非線形な「彫刻」学習則 """
        with torch.no_grad():
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 修正3: 報酬の非線形増幅 (Soft-Win)
            # 良い結果には指数関数的なご褒美、悪い結果には線形な罰
            if reward > 0:
                reward_gain = torch.exp(torch.tensor(reward)).item()
                self.states.add_(trace * reward_gain * 0.5)
            else:
                self.states.sub_(trace * abs(reward) * 0.1)
            
            # 修正4: 目標密度への「緩やかな引き込み」
            # 振動を抑えるため、変化率を conn_rate との差分で制御
            diff = self.target_conn_rate - conn_rate
            self.states.add_(diff * 0.5) # 定常的な成長/減衰の力

            self.states.clamp_(1, self.max_states)
