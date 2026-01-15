# snn_research/learning_rules/reward_modulated_stdp.py
# 修正: updateのシグネチャ変更と、戻り値を (delta_w, None) に変更

import torch
from typing import Optional, Any, Tuple, Dict
from snn_research.learning_rules.base_rule import BioLearningRule

class RewardModulatedSTDP(BioLearningRule):
    def __init__(self, learning_rate: float = 1e-4, time_window: int = 20, **kwargs: Any):
        super().__init__()
        self.lr = learning_rate
        self.window = time_window
        self.eligibility_trace: Optional[torch.Tensor] = None

    def update(
        self, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor, 
        weights: torch.Tensor, 
        optional_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Any]:
        
        # 報酬の取得 (optional_params または kwargs から)
        reward = 0.0
        if optional_params is not None and 'reward' in optional_params:
            reward = optional_params['reward']
        elif 'reward' in kwargs:
            reward = kwargs['reward']

        # STDP計算
        stdp_update = torch.matmul(post_spikes.t(), pre_spikes)
        
        if self.eligibility_trace is None:
            self.eligibility_trace = torch.zeros_like(weights)
        
        trace = self.eligibility_trace
        self.eligibility_trace = trace * 0.9 + stdp_update
        
        delta_w = self.lr * self.eligibility_trace * reward
        
        # simple_network.py は (dw, info) を期待しているため、変化量を返す
        return delta_w, None

class EmotionModulatedSTDP(RewardModulatedSTDP):
    def update(
        self, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor, 
        weights: torch.Tensor, 
        optional_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Any]:
        
        external_reward = 0.0
        internal_value = 0.0
        
        if optional_params:
            external_reward = optional_params.get('external_reward', 0.0)
            internal_value = optional_params.get('internal_value', 0.0)
        else:
            external_reward = kwargs.get('external_reward', 0.0)
            internal_value = kwargs.get('internal_value', 0.0)
        
        stdp_update = torch.matmul(post_spikes.t(), pre_spikes)
        
        if self.eligibility_trace is None:
            self.eligibility_trace = torch.zeros_like(weights)
        
        trace = self.eligibility_trace
        self.eligibility_trace = trace * 0.9 + stdp_update
        
        alpha = 1.0 
        modulation_signal = external_reward + alpha * internal_value
        
        delta_w = self.lr * self.eligibility_trace * modulation_signal
        
        return delta_w, None