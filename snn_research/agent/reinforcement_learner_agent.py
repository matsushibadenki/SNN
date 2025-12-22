# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# Title: 強化学習エージェント (BioSNN 引数整合性修正版)
# Description: BioSNNのリファクタリングに伴う引数不一致(Unexpected keyword argument)を解消。

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, cast

from snn_research.models.bio.simple_network import BioSNN
from snn_research.learning_rules.base_rule import BioLearningRule
from snn_research.communication import SpikeEncoderDecoder

class ReinforcementLearnerAgent:
    """
    BioSNNと報酬変調型STDPを用い、GRPOによる論理推論強化を行ったエージェント。
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
        
        hidden_size = (input_size + output_size) * 2
        layer_sizes = [input_size, hidden_size, output_size]
        
        # BioSNNのリファクタリング後の引数構成に適合させる
        self.model = BioSNN(
            layer_sizes=layer_sizes,
            neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0},
            synaptic_rule=synaptic_rule,
            homeostatic_rule=homeostatic_rule,
            neuron_type="adaptive_lif"  # デフォルト値を明示
        ).to(device)

        self.encoder = SpikeEncoderDecoder(num_neurons=input_size, time_steps=1)
        self.experience_buffer: List[List[torch.Tensor]] = []

    def get_action(self, state: torch.Tensor, record_experience: bool = True) -> int:
        self.model.eval()
        with torch.no_grad():
            if state.max() > 1.0 or state.dtype == torch.float32:
                 input_spikes = (torch.rand_like(state) < (state * 0.5 + 0.5)).float()
            else:
                 input_spikes = state

            output_spikes, hidden_spikes_history = self.model(input_spikes)
            
            if record_experience:
                # 入力スパイクと隠れ層の履歴を統合して保存
                self.experience_buffer.append([input_spikes] + hidden_spikes_history)
            
            if output_spikes.sum() > 0:
                action = torch.argmax(output_spikes).item()
            else:
                action = torch.randint(0, self.output_size, (1,)).item()
                
            return int(action)

    def learn(self, reward: float, causal_credit: float = 0.0, global_context: Optional[Dict[str, Any]] = None):
        if not self.experience_buffer:
            return

        self.model.train()
        
        if causal_credit > 0:
            final_reward_signal = reward + causal_credit * 10.0 
        else:
            final_reward_signal = reward
            
        optional_params: Dict[str, Any] = {"reward": final_reward_signal}
        
        if global_context:
            optional_params["global_workspace_context"] = global_context

        for step_spikes in self.experience_buffer:
            # BioSNN.update_weights(all_layer_spikes, optional_params) に適合
            self.model.update_weights(
                all_layer_spikes=step_spikes,
                optional_params=optional_params
            )
        
        self.experience_buffer = []

    def sample_thought_trajectories(
        self, 
        initial_state: torch.Tensor, 
        env_step_func: Callable[[int], Tuple[torch.Tensor, float, bool, Any]],
        num_samples: int = 4, 
        max_steps: int = 10
    ) -> List[Dict[str, Any]]:
        trajectories = []
        
        for _ in range(num_samples):
            current_state = initial_state.clone()
            
            trajectory: Dict[str, Any] = {
                'actions': [],
                'rewards': [],
                'spikes_history': [], 
                'total_reward': 0.0
            }
            
            self.experience_buffer = []
            
            for _ in range(max_steps):
                action = self.get_action(current_state, record_experience=True)
                next_state, reward, done, _ = env_step_func(action)
                
                cast(List[int], trajectory['actions']).append(action)
                cast(List[float], trajectory['rewards']).append(reward)
                
                current_total = cast(float, trajectory['total_reward'])
                trajectory['total_reward'] = current_total + reward
                
                current_state = next_state
                if done:
                    break
            
            trajectory['spikes_history'] = list(self.experience_buffer)
            trajectories.append(trajectory)
            
        self.experience_buffer = []
        return trajectories

    def learn_with_grpo(self, trajectories: List[Dict[str, Any]], baseline_reward: float = 0.0):
        if not trajectories:
            return

        self.model.train()
        
        total_rewards = torch.tensor([t['total_reward'] for t in trajectories], dtype=torch.float32)
        group_mean = total_rewards.mean()
        group_std = total_rewards.std() + 1e-8
        
        advantages = (total_rewards - group_mean) / group_std
        
        print(f"🧠 GRPO Update: Group Mean Reward: {group_mean:.4f}, Std: {group_std:.4f}")
        
        for i, trajectory in enumerate(trajectories):
            adv = advantages[i].item()
            adv = np.clip(adv, -2.0, 2.0)
            optional_params = {"reward": adv}
            
            history = cast(List[List[torch.Tensor]], trajectory['spikes_history'])
            
            for step_spikes in history:
                # 引数名を最新のBioSNNの定義に合わせる
                self.model.update_weights(
                    all_layer_spikes=step_spikes,
                    optional_params=optional_params
                )
