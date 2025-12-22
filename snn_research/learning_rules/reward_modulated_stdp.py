# ファイルパス: snn_research/learning_rules/reward_modulated_stdp.py
# Title: Reward Modulated STDP (Base for Causal Trace)

import torch
from typing import Dict, Any, Optional, Tuple
# 相対インポートで循環参照を回避しつつSTDPを取得
from .stdp import STDP

class RewardModulatedSTDP(STDP):
    """報酬によって変調される適格度トレースベースの学習則。"""
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, tau_eligibility: float, dt: float = 1.0):
        super().__init__(learning_rate, a_plus, a_minus, tau_trace, dt)
        self.tau_eligibility = tau_eligibility
        self.eligibility_trace: Optional[torch.Tensor] = None

    def _initialize_eligibility_trace(self, weight_shape: torch.Size, device: torch.device):
        self.eligibility_trace = torch.zeros(weight_shape, device=device)

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor, 
               optional_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # STDPの計算を利用して dw を potential_dw として取得
        potential_dw, _ = super().update(pre_spikes, post_spikes, weights, optional_params)
        
        if self.eligibility_trace is None or self.eligibility_trace.shape != weights.shape:
            self._initialize_eligibility_trace(weights.shape, weights.device)
            
        assert self.eligibility_trace is not None
        
        # 適格度トレースの更新
        self.eligibility_trace = self.eligibility_trace * (1.0 - self.dt / self.tau_eligibility) + potential_dw
        
        reward = optional_params.get("reward", 0.0) if optional_params else 0.0
        dw = reward * self.eligibility_trace
        
        # 次レイヤーへのクレジット伝播（簡易版）
        backward_credit = torch.matmul(dw.t(), post_spikes.mean(dim=0).unsqueeze(1)).squeeze() if post_spikes.dim() > 1 else torch.zeros_like(pre_spikes)
        
        return dw, backward_credit
