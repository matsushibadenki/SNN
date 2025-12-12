# ファイルパス: snn_research/learning_rules/reward_modulated_stdp.py
# Title: Reward Modulated STDP [Batch Support Fixed]
# Description:
#   バッチ入力に対応した報酬変調型STDP。

import torch
from typing import Dict, Any, Optional, Tuple
from .stdp import STDP

class RewardModulatedSTDP(STDP):
    """STDPと適格性トレース(Eligibility Trace)を用いた報酬ベースの学習ルール。"""
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, tau_eligibility: float, dt: float = 1.0):
        super().__init__(learning_rate, a_plus, a_minus, tau_trace, dt)
        self.tau_eligibility = tau_eligibility
        self.eligibility_trace: Optional[torch.Tensor] = None

    def _initialize_eligibility_trace(self, weight_shape: tuple, device: torch.device):
        self.eligibility_trace = torch.zeros(weight_shape, device=device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """報酬信号に基づいて重み変化量を計算する。"""
        
        if pre_spikes.dim() == 1: pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1: post_spikes = post_spikes.unsqueeze(0)

        if (self.pre_trace is None or self.pre_trace.shape != pre_spikes.shape):
            self._initialize_traces(pre_spikes.shape, post_spikes.shape, pre_spikes.device)
            
        self._update_traces(pre_spikes, post_spikes)
        
        if self.eligibility_trace is None or self.eligibility_trace.shape != weights.shape:
            self._initialize_eligibility_trace(weights.shape, weights.device)
        
        assert self.pre_trace is not None and self.post_trace is not None and self.eligibility_trace is not None

        # Eligibility Trace Update (Batch Summation)
        # LTP: Post * Pre_trace -> (Post, Pre)
        ltp = torch.matmul(post_spikes.t(), self.pre_trace)
        # LTD: Post_trace * Pre -> (Post, Pre)
        ltd = torch.matmul(self.post_trace.t(), pre_spikes)

        self.eligibility_trace += self.a_plus * ltp
        self.eligibility_trace -= self.a_minus * ltd
        self.eligibility_trace -= (self.eligibility_trace / self.tau_eligibility) * self.dt
        
        reward = optional_params.get("reward", 0.0) if optional_params else 0.0
        
        if reward != 0.0:
            dw = self.learning_rate * reward * self.eligibility_trace
            # 報酬適用後はトレースをリセット（ポリシーによるが、ここではリセットして次の報酬を待つ）
            self.eligibility_trace *= 0.0
            return dw, None
        
        return torch.zeros_like(weights), None