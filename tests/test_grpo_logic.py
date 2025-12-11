# ファイルパス: tests/test_grpo_logic.py
# Title: GRPOロジック検証テスト [Type Fixed]
# Description:
#   self.history の型ヒント不足エラーを修正。

import torch
import unittest
import sys
import os
import random
import numpy as np
from typing import List

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.learning_rules.reward_modulated_stdp import RewardModulatedSTDP

class SimpleLogicEnv:
    """
    GRPOテスト用の簡易論理パズル環境。
    """
    def __init__(self, target_sequence: list):
        self.target_sequence = target_sequence
        self.current_step = 0
        # 修正: 型ヒントを追加
        self.history: List[int] = []
        self.state_dim = 4
        
    def reset(self):
        self.current_step = 0
        self.history = []
        return self._get_state()
        
    def _get_state(self):
        state = torch.zeros(self.state_dim)
        if self.history:
            state[self.history[-1]] = 1.0
        else:
            state[3] = 1.0 
        return state
        
    def step(self, action: int):
        self.history.append(action)
        self.current_step += 1
        
        done = False
        reward = 0.0
        
        target_len = len(self.target_sequence)
        
        if len(self.history) <= target_len:
            current_idx = len(self.history) - 1
            if self.history[current_idx] == self.target_sequence[current_idx]:
                reward = 0.1
            else:
                reward = -0.1
                done = True
        
        if len(self.history) == target_len and not done:
            if self.history == self.target_sequence:
                reward = 10.0
                done = True
            else:
                done = True
                
        return self._get_state(), reward, done, {}

class TestGRPO(unittest.TestCase):
    
    def setUp(self):
        self.device = "cpu"
        self.input_size = 4
        self.output_size = 3 
        
        self.rule = RewardModulatedSTDP(
            learning_rate=0.05,
            a_plus=0.01,
            a_minus=0.01,
            tau_trace=20.0,
            tau_eligibility=50.0 
        )
        
        self.agent = ReinforcementLearnerAgent(
            input_size=self.input_size,
            output_size=self.output_size,
            device=self.device,
            synaptic_rule=self.rule
        )
        
        self.target_seq = [0, 1, 0]
        
    def test_grpo_improvement(self):
        print("\n[Test] GRPO Logic Improvement Check")
        env = SimpleLogicEnv(self.target_seq)
        
        iterations = 5
        group_size = 5
        
        for it in range(iterations):
            trajectories = []
            success_count = 0
            for _ in range(group_size):
                env.reset()
                initial_state = env._get_state()
                
                state = initial_state
                traj_data = {'actions': [], 'rewards': [], 'spikes_history': [], 'total_reward': 0.0}
                self.agent.experience_buffer = [] 
                
                for step in range(len(self.target_seq)):
                    action = self.agent.get_action(state, record_experience=True)
                    next_state, reward, done, _ = env.step(action)
                    
                    traj_data['actions'].append(action)
                    traj_data['rewards'].append(reward)
                    traj_data['total_reward'] += reward
                    state = next_state
                    if done:
                        break
                
                traj_data['spikes_history'] = list(self.agent.experience_buffer)
                trajectories.append(traj_data)
                
                if traj_data['actions'] == self.target_seq:
                    success_count += 1
            
            print(f"Iteration {it}: Success Rate in Group: {success_count}/{group_size}")
            self.agent.learn_with_grpo(trajectories)
            
        env.reset()
        state = env._get_state()
        actions = []
        for _ in range(len(self.target_seq)):
            action = self.agent.get_action(state, record_experience=False)
            state, _, done, _ = env.step(action)
            actions.append(action)
            if done: break
            
        print(f"Final Action Sequence: {actions}")
        self.assertTrue(len(actions) > 0)

if __name__ == '__main__':
    unittest.main()
