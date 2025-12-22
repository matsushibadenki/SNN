# ファイルパス: snn_research/learning_rules/stdp.py
# Title: STDP & TripletSTDP 学習則 (機能復元版)
# Description: 他のモジュールから参照される STDP クラスを復元し、mypyエラーを解消。

import torch
from typing import Dict, Any, Optional, Tuple, cast
from .base_rule import BioLearningRule

class STDP(BioLearningRule):
    """標準的なペアベースSTDP学習則。"""
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, dt: float = 1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_trace = tau_trace
        self.dt = dt
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor,
               optional_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.pre_trace is None or self.pre_trace.shape != pre_spikes.shape:
            self.pre_trace = torch.zeros_like(pre_spikes)
            self.post_trace = torch.zeros_like(post_spikes)

        assert self.pre_trace is not None and self.post_trace is not None

        with torch.no_grad():
            self.pre_trace = self.pre_trace * (1.0 - self.dt / self.tau_trace) + pre_spikes
            self.post_trace = self.post_trace * (1.0 - self.dt / self.tau_trace) + post_spikes

        dw = self.learning_rate * (self.a_plus * torch.matmul(post_spikes.t(), self.pre_trace) - 
                                  self.a_minus * torch.matmul(self.post_trace.t(), pre_spikes))
        return dw, None

class TripletSTDP(BioLearningRule):
    """動的適合型 Triplet STDP。"""
    def __init__(self, learning_rate: float = 0.01, target_rate: float = 0.05, dt: float = 1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.target_rate = target_rate
        self.dt = dt
        self.tau_plus = torch.tensor(16.8)
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None
        self.avg_firing_rate: Optional[torch.Tensor] = None

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor,
               optional_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.pre_trace is None:
            self.pre_trace = torch.zeros_like(pre_spikes)
            self.post_trace = torch.zeros_like(post_spikes)
            self.avg_firing_rate = torch.full((post_spikes.shape[-1],), self.target_rate, device=pre_spikes.device)

        activity_scale = torch.clamp(cast(torch.Tensor, self.avg_firing_rate) / self.target_rate, 0.5, 2.0)
        adj_tau_plus = self.tau_plus / activity_scale.mean()

        with torch.no_grad():
            self.pre_trace = self.pre_trace * (1.0 - self.dt / adj_tau_plus) + pre_spikes
            self.avg_firing_rate = 0.999 * cast(torch.Tensor, self.avg_firing_rate) + 0.001 * post_spikes.mean(dim=0)
        
        reward = optional_params.get("reward", 1.0) if optional_params else 1.0
        dw = self.learning_rate * reward * torch.matmul(post_spikes.t(), self.pre_trace)
        return dw, None
