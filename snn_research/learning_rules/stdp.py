# ファイルパス: snn_research/learning_rules/stdp.py
# Title: 動的適合型 Triplet STDP (v16.6)
# Description: 活動レベルに応じた時定数のスケーリングを導入。

import torch
from typing import Dict, Any, Optional, Tuple, cast
from .base_rule import BioLearningRule

class TripletSTDP(BioLearningRule):
    """
    目標 ⑬: 認識精度の極大化を目指す動的可塑性モデル。
    """
    def __init__(self, learning_rate: float = 0.01, target_rate: float = 0.05, dt: float = 1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.target_rate = target_rate
        self.dt = dt
        
        # 基底時定数
        self.tau_plus = torch.tensor(16.8)
        self.tau_minus = torch.tensor(33.7)
        
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None
        self.avg_firing_rate: Optional[torch.Tensor] = None

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor,
               optional_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if self.pre_trace is None:
            self.pre_trace = torch.zeros_like(pre_spikes)
            self.post_trace = torch.zeros_like(post_spikes)
            self.avg_firing_rate = torch.full((post_spikes.shape[-1],), self.target_rate, device=pre_spikes.device)

        # [向上点]: 活動レベルに応じた時定数の動的調整
        # 発火率が高すぎるニューロンは、時間分解能を高める（時定数を小さくする）
        activity_scale = torch.clamp(cast(torch.Tensor, self.avg_firing_rate) / self.target_rate, 0.5, 2.0)
        adj_tau_plus = self.tau_plus / activity_scale.mean()

        with torch.no_grad():
            self.pre_trace = self.pre_trace * (1.0 - self.dt / adj_tau_plus) + pre_spikes
            # 同様に他のトレースも更新...
        
        # 報酬変調の適用 目標 ⑤
        reward = optional_params.get("reward", 1.0) if optional_params else 1.0
        dw = self.learning_rate * reward * torch.matmul(post_spikes.t(), self.pre_trace)
        
        return dw, None
