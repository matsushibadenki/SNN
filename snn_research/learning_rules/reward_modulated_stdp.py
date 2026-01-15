# snn_research/learning_rules/reward_modulated_stdp.py
# 修正: BioLearningRuleを継承

import torch
from typing import Optional
from snn_research.learning_rules.base_rule import BioLearningRule

class RewardModulatedSTDP(BioLearningRule):
    # ... 中身はそのまま ...
    def __init__(self, learning_rate: float = 1e-4, time_window: int = 20, **kwargs):
        self.lr = learning_rate
        self.window = time_window
        self.eligibility_trace: Optional[torch.Tensor] = None

    def update(self, weights: torch.Tensor, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0, **kwargs):
        # ... 中身はそのまま ...
        stdp_update = torch.matmul(post_spikes.t(), pre_spikes)
        
        if self.eligibility_trace is None:
            self.eligibility_trace = torch.zeros_like(weights)
        
        trace = self.eligibility_trace
        self.eligibility_trace = trace * 0.9 + stdp_update
        
        delta_w = self.lr * self.eligibility_trace * reward
        
        return weights + delta_w

# EmotionModulatedSTDP も同様にそのまま (RewardModulatedSTDPを継承しているのでOK)
class EmotionModulatedSTDP(RewardModulatedSTDP):
    def update(self, weights: torch.Tensor, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, 
               external_reward: float = 0.0, internal_value: float = 0.0, **kwargs):
        # ... 中身はそのまま ...
        stdp_update = torch.matmul(post_spikes.t(), pre_spikes)
        
        if self.eligibility_trace is None:
            self.eligibility_trace = torch.zeros_like(weights)
        
        trace = self.eligibility_trace
        self.eligibility_trace = trace * 0.9 + stdp_update
        
        alpha = 1.0 
        modulation_signal = external_reward + alpha * internal_value
        
        delta_w = self.lr * self.eligibility_trace * modulation_signal
        
        return weights + delta_w