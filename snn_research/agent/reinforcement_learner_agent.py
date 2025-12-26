# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# Title: RL Agent (Temperature 4.0 - Optimized Exploitation)
# Description: ノイズ除去による信号品質向上に伴い、探索パラメータを最適化。

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, cast

from snn_research.core.hybrid_core import HybridNeuromorphicCore
from snn_research.learning_rules.base_rule import BioLearningRule

class ReinforcementLearnerAgent:
    """
    SCAL (Statistical Centroid Alignment Learning) を搭載した次世代強化学習エージェント。
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        device: str, 
        synaptic_rule: Optional[BioLearningRule] = None, 
        homeostatic_rule: Optional[BioLearningRule] = None
    ):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        
        hidden_dim = 64
        
        self.model = HybridNeuromorphicCore(
            in_features=input_size,
            hidden_features=hidden_dim,
            out_features=output_size
        ).to(device)

        self.experience_buffer: List[Dict[str, torch.Tensor]] = []

    def get_action(self, state: torch.Tensor, record_experience: bool = True) -> int:
        self.model.eval() 
        
        if record_experience:
            self.model.train()
        
        with torch.no_grad():
            f = self.model.fast_process(state)
            r = self.model.deep_process(f)
            out = self.model.output_gate(r)
            
            if self.model.training:
                # [修正] 信号品質向上により、温度を 4.0 に調整して収束を早める
                probs = torch.softmax(out / 4.0, dim=1)
            else:
                probs = torch.softmax(out, dim=1)
            
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / self.output_size
            
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            action_idx = int(action.item())

            if record_experience:
                step_data = {
                    'pre_spikes': r.clone(), 
                    'action': action_idx,
                    'probs': probs.clone()
                }
                self.experience_buffer.append(step_data)
            
            return action_idx

    def learn(self, reward: float, causal_credit: float = 0.0, global_context: Optional[Dict[str, Any]] = None):
        pass 

    def learn_with_grpo(self, trajectories: List[Dict[str, Any]], baseline_reward: float = 0.0):
        if not trajectories:
            return

        self.model.train()
        
        total_rewards = torch.tensor([t['total_reward'] for t in trajectories], dtype=torch.float32)
        
        if len(total_rewards) > 1:
            mean_reward = total_rewards.mean()
            std_reward = total_rewards.std() + 1e-8
            advantages = (total_rewards - mean_reward) / std_reward
        else:
            advantages = torch.zeros_like(total_rewards)
        
        for i, trajectory in enumerate(trajectories):
            self.model.reset_state()
            
            adv = float(advantages[i].item())
            clipped_reward = float(np.clip(adv, -1.0, 1.0))
            
            episode_history = trajectory.get('spikes_history', [])
            
            for step_data in episode_history:
                if not isinstance(step_data, dict): 
                    continue 

                pre_spikes = step_data['pre_spikes'] 
                action_idx = step_data['action']
                
                reward_signal = torch.zeros((1, self.output_size), device=self.device)
                reward_signal[0, action_idx] = clipped_reward
                
                post_spikes_target = torch.zeros((1, self.output_size), device=self.device)
                post_spikes_target[0, action_idx] = 1.0
                
                self.model.output_gate.update_plasticity(
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes_target, 
                    reward=reward_signal, 
                    learning_rate=0.05
                )
