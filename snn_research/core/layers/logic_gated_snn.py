# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (三値・安定適応版)
# 目的: 結合率の激しい変動を抑え、報酬に基づいた「意味のあるスパース性」を固定する。

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
        
        # 初期状態: 閾値付近でガウス分布させ、多様性を持たせる
        states = torch.randn(out_features, in_features) * 2.0 + self.threshold
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        """ 1.58ビット (-1, 0, 1) を模倣した重み。本実装では 0/1 に固定。"""
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)

        # 電流計算 (デジタル累積)
        current = torch.matmul(x, w.t()).view(-1)
        
        # 膜電位更新
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.copy_(v_mem * 0.8 + current) 
        
        # 発火判定
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 適格性トレースの更新 (行列演算なし: outer product)
        with torch.no_grad():
            # プレとポストの共起を記録 (時間定数を長くする)
            self.eligibility_trace.mul_(0.95).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.5)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """報酬に基づく微細な配線調整"""
        with torch.no_grad():
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # 修正1: 更新の「歩幅」を小さくし、振動を抑える (0.1 単位)
            # reward のスケーリングをマイルドにする
            stable_reward = torch.tanh(torch.tensor(reward / 10.0)).item()
            
            # 修正2: LTP/LTD の適用
            update = trace * stable_reward * 2.0
            self.states.add_(update)
            
            # 修正3: 自律的なスパース性維持 (強すぎない剪定)
            # 常に微弱な「忘却」を入れ、新しい結合の席を空ける
            self.states.sub_(0.01)

            # 修正4: 死滅防止 (最低限の好奇心)
            conn_rate = self.get_ternary_weights().mean().item()
            if conn_rate < 0.05:
                # 結合が少なすぎる場合、ランダムに少数を Include 状態の予備軍へ
                revive_mask = torch.rand_like(self.states) < 0.001
                self.states[revive_mask] += 5.0

            self.states.clamp_(1, self.max_states)
