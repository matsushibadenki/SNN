# ファイルパス: snn_research/learning_rules/stdp.py
# Title: STDP & TripletSTDP Learning Rules (Mypy Fixed)
# Description: バッチ対応と型安全性を確保した高精度可塑性モデル。

import torch
from typing import Dict, Any, Optional, Tuple, cast
from .base_rule import BioLearningRule

class STDP(BioLearningRule):
    """ペアベースのSTDP学習則。"""
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, dt: float = 1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_trace = tau_trace
        self.dt = dt
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None

    def _initialize_traces(self, pre_shape: torch.Size, post_shape: torch.Size, device: torch.device):
        self.pre_trace = torch.zeros(pre_shape, device=device)
        self.post_trace = torch.zeros(post_shape, device=device)

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor, 
               optional_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if pre_spikes.dim() == 1: pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1: post_spikes = post_spikes.unsqueeze(0)

        if self.pre_trace is None or self.pre_trace.shape != pre_spikes.shape:
            self._initialize_traces(pre_spikes.shape, post_spikes.shape, pre_spikes.device)

        assert self.pre_trace is not None and self.post_trace is not None

        with torch.no_grad():
            self.pre_trace = self.pre_trace * (1.0 - self.dt / self.tau_trace) + pre_spikes
            self.post_trace = self.post_trace * (1.0 - self.dt / self.tau_trace) + post_spikes

        ltp = torch.matmul(post_spikes.t(), self.pre_trace)
        ltd = torch.matmul(self.post_trace.t(), pre_spikes)
        dw = self.learning_rate * (self.a_plus * ltp - self.a_minus * ltd)
        return dw, None

class TripletSTDP(BioLearningRule):
    """報酬変調型 Triplet STDP。Mypyエラーを解消し、Objective.mdの精度目標を反映。"""
    def __init__(self, learning_rate: float = 0.01, target_rate: float = 0.05, 
                 homeostasis_strength: float = 0.2, dt: float = 1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.target_rate = target_rate
        self.homeostasis_strength = homeostasis_strength
        self.dt = dt
        
        # 生理学的時定数
        self.tau_plus, self.tau_minus = 16.8, 33.7
        self.tau_x, self.tau_y = 101.0, 125.0
        self.a2_plus, self.a2_minus = 7.5e-3, 7.0e-3
        self.a3_plus, self.a3_minus = 9.3e-3, 2.3e-4

        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None
        self.pre_trace_triplet: Optional[torch.Tensor] = None
        self.post_trace_triplet: Optional[torch.Tensor] = None
        self.avg_firing_rate: Optional[torch.Tensor] = None

    def _initialize_traces(self, pre_shape: torch.Size, post_shape: torch.Size, device: torch.device):
        self.pre_trace = torch.zeros(pre_shape, device=device)
        self.post_trace = torch.zeros(post_shape, device=device)
        self.pre_trace_triplet = torch.zeros(pre_shape, device=device)
        self.post_trace_triplet = torch.zeros(post_shape, device=device)
        self.avg_firing_rate = torch.full((post_shape[1],), self.target_rate, device=device)

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor,
               optional_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if pre_spikes.dim() == 1: pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1: post_spikes = post_spikes.unsqueeze(0)

        if self.pre_trace is None or self.pre_trace.shape != pre_spikes.shape:
            self._initialize_traces(pre_spikes.shape, post_spikes.shape, pre_spikes.device)

        # Mypy対策: 存在を保証
        pre_t = cast(torch.Tensor, self.pre_trace)
        post_t = cast(torch.Tensor, self.post_trace)
        pre_t3 = cast(torch.Tensor, self.pre_trace_triplet)
        post_t3 = cast(torch.Tensor, self.post_trace_triplet)
        avg_r = cast(torch.Tensor, self.avg_firing_rate)

        with torch.no_grad():
            self.pre_trace = pre_t * (1.0 - self.dt / self.tau_plus) + pre_spikes
            self.post_trace = post_t * (1.0 - self.dt / self.tau_minus) + post_spikes
            self.pre_trace_triplet = pre_t3 * (1.0 - self.dt / self.tau_x) + pre_spikes
            self.post_trace_triplet = post_t3 * (1.0 - self.dt / self.tau_y) + post_spikes
            self.avg_firing_rate = 0.999 * avg_r + 0.001 * post_spikes.mean(dim=0)

        reward = optional_params.get("reward", 1.0) if optional_params else 1.0
        
        dw_2_plus = torch.matmul(post_spikes.t(), cast(torch.Tensor, self.pre_trace))
        dw_2_minus = torch.matmul(cast(torch.Tensor, self.post_trace).t(), pre_spikes)
        dw_3_plus = torch.matmul(post_spikes.t(), cast(torch.Tensor, self.pre_trace_triplet))
        
        dw = self.a2_plus * dw_2_plus + self.a3_plus * dw_3_plus - self.a2_minus * dw_2_minus
        
        # 恒常性変調
        rate_error = (cast(torch.Tensor, self.avg_firing_rate) - self.target_rate) * self.homeostasis_strength
        homeostasis_mod = torch.clamp(1.0 - rate_error.unsqueeze(1), min=0.0, max=2.0)
        
        return self.learning_rate * reward * dw * homeostasis_mod, None
