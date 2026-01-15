# snn_research/learning_rules/stdp.py
# Title: Advanced STDP with Homeostasis (Phase 2 Optimized)
# Description: 
#   Vectorized STDPに、Synaptic Scaling (Homeostasis) と Soft Bound を追加。
#   学習安定性 95% 目標に向けた強化版。
#   [Fix] BioLearningRuleのシグネチャ変更に対応 (optional_params追加)

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict, Any
from .base_rule import BioLearningRule

class STDP(nn.Module, BioLearningRule):
    """
    Vectorized Spike-Timing Dependent Plasticity (STDP) with Homeostasis.
    Phase 2 Features:
    - Soft Bound Weight Update
    - Synaptic Scaling (Normalization)
    - Heterosynaptic Plasticity (Decay)
    """

    def __init__(
        self,
        learning_rate: Union[float, Tuple[float, float]] = (1e-4, -1e-4),
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        dt: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        enable_homeostasis: bool = True,
        homeostasis_rate: float = 0.001,
        **kwargs: Any
    ):
        super().__init__()
        if isinstance(learning_rate, float):
            self.lr_ltp = learning_rate
            self.lr_ltd = -learning_rate * 0.8  # LTDは少し弱めにするのが一般的
        else:
            self.lr_ltp = learning_rate[0]
            self.lr_ltd = learning_rate[1]

        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.dt = dt
        self.w_min = w_min
        self.w_max = w_max
        
        # Homeostasis config
        self.enable_homeostasis = enable_homeostasis
        self.homeostasis_rate = homeostasis_rate

    def forward(
        self,
        weight: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        pre_trace: Optional[torch.Tensor] = None,
        post_trace: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if pre_spikes.dim() == 2:
            pre_spikes = pre_spikes.unsqueeze(0)
            post_spikes = post_spikes.unsqueeze(0)

        T, B, N_in = pre_spikes.shape
        _, _, N_out = post_spikes.shape
        device = weight.device

        if pre_trace is None:
            pre_trace = torch.zeros(B, N_in, device=device)
        if post_trace is None:
            post_trace = torch.zeros(B, N_out, device=device)

        decay_pre = torch.exp(torch.tensor(-self.dt / self.tau_pre, device=device))
        decay_post = torch.exp(torch.tensor(-self.dt / self.tau_post, device=device))

        delta_w = torch.zeros_like(weight)
        
        curr_pre_trace = pre_trace
        curr_post_trace = post_trace

        for t in range(T):
            pre_s = pre_spikes[t]
            post_s = post_spikes[t]

            # Trace Update
            curr_pre_trace = curr_pre_trace * decay_pre + pre_s
            curr_post_trace = curr_post_trace * decay_post + post_s

            # STDP Calculation
            ltp = torch.bmm(post_s.unsqueeze(2), curr_pre_trace.unsqueeze(1))
            ltd = torch.bmm(curr_post_trace.unsqueeze(2), pre_s.unsqueeze(1))
            
            # Batch mean
            ltp_mean = ltp.mean(dim=0)
            ltd_mean = ltd.mean(dim=0)

            # Soft Bound Scaling
            soft_bound_ltp = (self.w_max - weight) if self.w_max > 0 else 1.0
            soft_bound_ltd = (weight - self.w_min) if self.w_min < 0 else weight

            delta_w += (self.lr_ltp * soft_bound_ltp * ltp_mean + 
                        self.lr_ltd * soft_bound_ltd * ltd_mean)

        return delta_w, curr_pre_trace, curr_post_trace

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        delta_w, _, _ = self.forward(weights, pre_spikes, post_spikes)
        
        # Heterosynaptic Plasticity / Synaptic Scaling
        if self.enable_homeostasis:
            total_activity = post_spikes.mean()
            decay_factor = self.homeostasis_rate * total_activity
            delta_w -= decay_factor * weights

        return delta_w, None

    def update_weight(self, weight: torch.Tensor, delta_w: torch.Tensor) -> torch.Tensor:
        new_weight = weight + delta_w
        return torch.clamp(new_weight, self.w_min, self.w_max)


class TripletSTDP(STDP):
    """Placeholder for Triplet STDP."""
    pass