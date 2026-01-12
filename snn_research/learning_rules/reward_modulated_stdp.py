# snn_research/learning_rules/reward_modulated_stdp.py
# Title: Reward Modulated STDP (Signature Compatible)
# Description: 親クラス STDP および BioLearningRule と整合するメソッド署名に修正。

import torch
from typing import Optional, Tuple, Dict, Any, Union
from .stdp import STDP

class RewardModulatedSTDP(STDP):
    """
    Reward-Modulated STDP.
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        a_plus: float = 1.0,
        a_minus: float = 1.0,
        tau_trace: float = 20.0,
        dt: float = 1.0,
        **kwargs: Any
    ):
        super().__init__(
            learning_rate=(learning_rate * a_plus, -learning_rate * a_minus),
            tau_pre=tau_trace,
            tau_post=tau_trace,
            dt=dt,
            **kwargs
        )

    def forward(
        self,
        weight: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        pre_trace: Optional[torch.Tensor] = None,
        post_trace: Optional[torch.Tensor] = None,
        reward: float = 0.0  # 親クラスの引数順序を守り、追加引数は末尾へ
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 1. 基本的なSTDP更新量を計算
        delta_w_stdp, new_pre, new_post = super().forward(
            weight, pre_spikes, post_spikes, pre_trace, post_trace
        )

        # 2. 報酬による変調
        delta_w = delta_w_stdp * reward

        return delta_w, new_pre, new_post
    
    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        辞書から reward を取得し、forward を呼び出す。
        """
        params = optional_params or {}
        reward = float(params.get("reward", 1.0)) # デフォルト報酬 1.0
        
        # forward呼び出し (traceは管理しない簡易版)
        delta_w, _, _ = self.forward(
            weight=weights,
            pre_spikes=pre_spikes, 
            post_spikes=post_spikes, 
            reward=reward
        )
        return delta_w, None