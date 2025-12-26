# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# Title: RL Agent (Correct GRPO Loop)
# Description: GRPO学習時に軌跡ごとに学習則をリセットし、干渉を防ぐ。

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, cast

from snn_research.models.bio.simple_network import BioSNN
from snn_research.learning_rules.base_rule import BioLearningRule

class ReinforcementLearnerAgent:
    """
    BioSNNを用いた強化学習エージェント。
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        device: str, 
        synaptic_rule: BioLearningRule, 
        homeostatic_rule: Optional[BioLearningRule] = None
    ):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        
        # 隠れ層設定
        hidden_dim = 64 
        layer_sizes = [input_size, hidden_dim, output_size]
        
        # ニューロンパラメータ
        neuron_params = {
            'tau_mem': 20.0, 
            'v_threshold': 0.5, 
            'v_reset': 0.0, 
            'dt': 1.0,
            'noise_std': 0.2
        }
        
        self.model = BioSNN(
            layer_sizes=layer_sizes,
            neuron_params=neuron_params,
            synaptic_rule=synaptic_rule,
            homeostatic_rule=homeostatic_rule,
            neuron_type="adaptive_lif"
        ).to(device)

        self.experience_buffer: List[List[torch.Tensor]] = []

    def get_action(self, state: torch.Tensor, record_experience: bool = True) -> int:
        self.model.eval() 
        
        if record_experience:
            self.model.train()
        
        with torch.no_grad():
            # 入力エンコーディング: 確率的なゆらぎを持たせる
            prob_input = state * 0.95 + 0.02
            input_spikes = (torch.rand_like(state) < prob_input).float()

            if input_spikes.dim() == 1:
                input_spikes = input_spikes.unsqueeze(0)

            output_spikes, hidden_history = self.model(input_spikes)
            
            if output_spikes.sum() > 0:
                action_idx = int(torch.argmax(output_spikes).item())
            else:
                action_idx = int(torch.randint(0, self.output_size, (1,)).item())

            if record_experience:
                final_history = [h.clone() for h in hidden_history]
                target_output = torch.zeros_like(output_spikes)
                if output_spikes.shape[1] > action_idx:
                     target_output[0, action_idx] = 1.0
                
                if len(final_history) > 0:
                    final_history[-1] = target_output
                
                self.experience_buffer.append(final_history)
            
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
            # [修正] 新しい軌跡（エピソード）の学習を始める前に、必ず学習則の内部状態をリセットする
            self.model.reset_learning_rules()

            adv = float(advantages[i].item())
            clipped_reward = float(np.clip(adv, -1.0, 1.0))
            optional_params = {"reward": clipped_reward}
            
            for step_spikes in cast(List[List[torch.Tensor]], trajectory['spikes_history']):
                self.model.update_weights(
                    all_layer_spikes=step_spikes,
                    optional_params=optional_params
                )
