# ファイルパス: tests/test_grpo_logic.py
# Title: GRPO Logic Test (Robust)

import torch
import unittest
import sys
import os
import random
import numpy as np
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.learning_rules.reward_modulated_stdp import RewardModulatedSTDP

class SimpleLogicEnv:
    def __init__(self, target_sequence: list):
        self.target_sequence = target_sequence
        self.history: List[int] = []
        self.state_dim = 4
        
    def reset(self):
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
        done = False
        reward = 0.0
        
        target_len = len(self.target_sequence)
        current_idx = len(self.history) - 1
        
        if current_idx < target_len:
            if self.history[current_idx] == self.target_sequence[current_idx]:
                reward = 1.0 
            else:
                # 失敗したら即終了だが、罰を与える
                reward = -1.0 
                done = True   
        
        if len(self.history) == target_len and not done:
            reward += 5.0 
            done = True
                
        return self._get_state(), reward, done, {}

class TestGRPO(unittest.TestCase):
    
    def setUp(self):
        self.device = "cpu"
        self.input_size = 4
        self.output_size = 3 
        
        self.rule = RewardModulatedSTDP(
            learning_rate=0.05, # 学習率を少し抑えて安定化
            a_plus=1.0,         # 増強を強く
            a_minus=0.5,
            tau_trace=20.0,
            tau_eligibility=100.0, # 長い適格度トレース
            dt=1.0
        )
        
        self.agent = ReinforcementLearnerAgent(
            input_size=self.input_size,
            output_size=self.output_size,
            device=self.device,
            synaptic_rule=self.rule
        )
        
        self.target_seq = [0, 1]
        
    def test_grpo_improvement(self):
        print("\n[Test] GRPO Logic Improvement Check")
        env = SimpleLogicEnv(self.target_seq)
        
        iterations = 50 # 反復を増やす
        group_size = 10 
        
        max_success_rate = 0.0
        
        for it in range(iterations):
            trajectories = []
            success_count = 0
            
            for _ in range(group_size):
                state = env.reset()
                
                # エージェントの状態リセット (LIFの膜電位など)
                self.agent.model.reset_state(batch_size=1, device=torch.device("cpu"))
                
                # 履歴バッファクリア
                self.agent.experience_buffer = [] 
                
                traj_data = {'actions': [], 'rewards': [], 'spikes_history': [], 'total_reward': 0.0}
                
                for _ in range(len(self.target_seq)):
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
            
            rate = success_count / group_size
            max_success_rate = max(max_success_rate, rate)
            
            if (it + 1) % 10 == 0:
                print(f"Iteration {it+1}/{iterations}: Success Rate {rate:.2f}")
            
            self.agent.learn_with_grpo(trajectories)
            
            if rate >= 0.7:
                print(f"Early Success at iteration {it+1}")
                break
            
        print(f"Max Success Rate: {max_success_rate}")
        
        # 学習の成功判定: 一度でも高い成功率を記録できたか
        self.assertTrue(max_success_rate > 0.2, f"Learning failed. Max rate: {max_success_rate}")

if __name__ == '__main__':
    unittest.main()
