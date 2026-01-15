# snn_research/learning_rules/causal_trace.py
# 修正: updateメソッドの引数順序と型を親クラス(RewardModulatedSTDP)に準拠させる。

import torch
from typing import Dict, Any, Optional
from snn_research.learning_rules.reward_modulated_stdp import RewardModulatedSTDP

class CausalTraceCreditAssignmentEnhancedV2(RewardModulatedSTDP):
    """
    因果トレースを用いた高度な信用割当学習則。
    """
    def __init__(self, 
                 learning_rate: float = 0.01, 
                 a_plus: float = 0.01, 
                 a_minus: float = 0.008, 
                 tau_trace: float = 20.0, 
                 tau_eligibility: float = 100.0,
                 **kwargs):
        
        # 親クラス初期化
        super().__init__(learning_rate=learning_rate, **kwargs)
        
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_trace = tau_trace
        self.tau_eligibility = tau_eligibility
        
        # 追加の内部状態
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None

    def update(self, 
               weights: torch.Tensor, 
               pre_spikes: torch.Tensor, 
               post_spikes: torch.Tensor, 
               reward: float = 0.0) -> torch.Tensor:
        """
        親クラスとシグネチャを統一: (weights, pre, post, reward)
        """
        batch_size = pre_spikes.size(0)

        # トレース初期化
        if self.pre_trace is None:
            self.pre_trace = torch.zeros_like(pre_spikes)
        if self.post_trace is None:
            self.post_trace = torch.zeros_like(post_spikes)
        if self.eligibility_trace is None:
            self.eligibility_trace = torch.zeros_like(weights)

        # トレース更新 (指数減衰 + スパイク)
        # decay factors
        alpha_pre = torch.exp(torch.tensor(-1.0 / self.tau_trace))
        alpha_post = torch.exp(torch.tensor(-1.0 / self.tau_trace))
        alpha_eligibility = torch.exp(torch.tensor(-1.0 / self.tau_eligibility))

        self.pre_trace = self.pre_trace * alpha_pre + pre_spikes
        self.post_trace = self.post_trace * alpha_post + post_spikes

        # STDP計算: Pre * Post_trace - Post * Pre_trace (簡易版)
        # (Out, In)
        potentiate = torch.matmul(post_spikes.t(), self.pre_trace)
        depress = torch.matmul(self.post_trace.t(), pre_spikes)
        
        stdp_update = self.a_plus * potentiate - self.a_minus * depress
        
        # 適合度トレース更新
        self.eligibility_trace = self.eligibility_trace * alpha_eligibility + stdp_update
        
        # 報酬による重み更新
        delta_w = self.lr * self.eligibility_trace * reward
        
        return weights + delta_w