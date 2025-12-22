# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# Title: 強化学習エージェント (BioSNN インターフェース完全適合版)
# Description: mypyの Unexpected keyword argument エラーを解消するため、シグネチャを完全同期。

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, cast

from snn_research.models.bio.simple_network import BioSNN
from snn_research.learning_rules.base_rule import BioLearningRule
from snn_research.communication import SpikeEncoderDecoder

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
        
        layer_sizes = [input_size, (input_size + output_size) * 2, output_size]
        
        # BioSNNの__init__に完全に一致させる
        self.model = BioSNN(
            layer_sizes=layer_sizes,
            neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0},
            synaptic_rule=synaptic_rule,
            homeostatic_rule=homeostatic_rule,
            neuron_type="adaptive_lif"
        ).to(device)

        self.experience_buffer: List[List[torch.Tensor]] = []

    def get_action(self, state: torch.Tensor, record_experience: bool = True) -> int:
        self.model.eval()
        with torch.no_grad():
            input_spikes = state if state.dtype == torch.float32 and state.max() <= 1.0 and state.min() >= 0.0 else (torch.rand_like(state) < (state * 0.5 + 0.5)).float()
            output_spikes, hidden_history = self.model(input_spikes)
            if record_experience:
                self.experience_buffer.append([input_spikes] + hidden_history)
            return int(torch.argmax(output_spikes).item()) if output_spikes.sum() > 0 else torch.randint(0, self.output_size, (1,)).item()

    def learn(self, reward: float, causal_credit: float = 0.0, global_context: Optional[Dict[str, Any]] = None):
        if not self.experience_buffer: return
        self.model.train()
        optional_params: Dict[str, Any] = {"reward": reward + causal_credit * 10.0}
        if global_context: optional_params["global_workspace_context"] = global_context

        for step_spikes in self.experience_buffer:
            # キーワード引数名をBioSNN.update_weightsの定義と完全に一致させる
            self.model.update_weights(
                all_layer_spikes=step_spikes,
                optional_params=optional_params
            )
        self.experience_buffer = []

    def sample_thought_trajectories(self, initial_state: torch.Tensor, env_step_func: Callable[[int], Tuple[torch.Tensor, float, bool, Any]], num_samples: int = 4, max_steps: int = 10) -> List[Dict[str, Any]]:
        trajectories = []
        for _ in range(num_samples):
            current_state = initial_state.clone()
            trajectory: Dict[str, Any] = {'actions': [], 'rewards': [], 'spikes_history': [], 'total_reward': 0.0}
            self.experience_buffer = []
            for _ in range(max_steps):
                action = self.get_action(current_state, record_experience=True)
                next_state, reward, done, _ = env_step_func(action)
                cast(List[int], trajectory['actions']).append(action)
                cast(List[float], trajectory['rewards']).append(reward)
                trajectory['total_reward'] = cast(float, trajectory['total_reward']) + reward
                current_state = next_state
                if done: break
            trajectory['spikes_history'] = list(self.experience_buffer)
            trajectories.append(trajectory)
        return trajectories

    def learn_with_grpo(self, trajectories: List[Dict[str, Any]], baseline_reward: float = 0.0):
        if not trajectories: return
        self.model.train()
        total_rewards = torch.tensor([t['total_reward'] for t in trajectories], dtype=torch.float32)
        advantages = (total_rewards - total_rewards.mean()) / (total_rewards.std() + 1e-8)
        
        for i, trajectory in enumerate(trajectories):
            optional_params = {"reward": np.clip(advantages[i].item(), -2.0, 2.0)}
            for step_spikes in cast(List[List[torch.Tensor]], trajectory['spikes_history']):
                # ここもキーワード引数を同期
                self.model.update_weights(
                    all_layer_spikes=step_spikes,
                    optional_params=optional_params
                )
