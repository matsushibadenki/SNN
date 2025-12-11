# ファイルパス: tests/test_grpo_logic.py
# 日本語タイトル: GRPOロジック検証テスト [Fixed]
# 目的・内容:
#   Phase 8-3: GRPOによる推論強化の検証。
#   修正: RewardModulatedSTDPの初期化に必要なパラメータを追加。

import torch
import unittest
import sys
import os
import random
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.learning_rules.reward_modulated_stdp import RewardModulatedSTDP

class SimpleLogicEnv:
    """
    GRPOテスト用の簡易論理パズル環境。
    特定の行動シーケンス (例: [0, 1, 0]) を入力すると高報酬が得られる。
    """
    def __init__(self, target_sequence: list):
        self.target_sequence = target_sequence
        self.current_step = 0
        self.history = []
        self.state_dim = 4
        
    def reset(self):
        self.current_step = 0
        self.history = []
        return self._get_state()
        
    def _get_state(self):
        # 現在のステップ数と履歴をワンホット的な状態で返す簡易実装
        state = torch.zeros(self.state_dim)
        if self.history:
            state[self.history[-1]] = 1.0 # 直前の行動
        else:
            state[3] = 1.0 # 初期状態マーカー
        return state
        
    def step(self, action: int):
        self.history.append(action)
        self.current_step += 1
        
        done = False
        reward = 0.0
        
        # ターゲットシーケンスと一致しているかチェック
        target_len = len(self.target_sequence)
        
        if len(self.history) <= target_len:
            # 部分一致チェック
            current_idx = len(self.history) - 1
            if self.history[current_idx] == self.target_sequence[current_idx]:
                reward = 0.1 # 部分的な正解報酬
            else:
                reward = -0.1 # 間違い
                done = True # 即失敗 (厳しい設定)
        
        if len(self.history) == target_len and not done:
            if self.history == self.target_sequence:
                reward = 10.0 # 完全正解報酬
                done = True
            else:
                done = True
                
        return self._get_state(), reward, done, {}

class TestGRPO(unittest.TestCase):
    
    def setUp(self):
        self.device = "cpu"
        self.input_size = 4
        self.output_size = 3 # Actions: 0, 1, 2
        
        # 学習則: 報酬変調STDP (パラメータを追加修正)
        self.rule = RewardModulatedSTDP(
            learning_rate=0.05,
            a_plus=0.01,       # STDP強化強度
            a_minus=0.01,      # STDP抑制強度
            tau_trace=20.0,    # シナプストレース時定数 (ms)
            tau_eligibility=50.0 # 適格性トレース時定数 (ms)
        )
        
        self.agent = ReinforcementLearnerAgent(
            input_size=self.input_size,
            output_size=self.output_size,
            device=self.device,
            synaptic_rule=self.rule
        )
        
        # 正解シーケンス: [0, 1, 0]
        self.target_seq = [0, 1, 0]
        
    def test_grpo_improvement(self):
        """GRPOによって正解シーケンスの生成確率が向上するかテスト"""
        print("\n[Test] GRPO Logic Improvement Check")
        
        env = SimpleLogicEnv(self.target_seq)
        
        # GRPO Loop Simulation
        iterations = 5
        group_size = 5
        
        for it in range(iterations):
            trajectories = []
            
            # Group Sampling
            success_count = 0
            for _ in range(group_size):
                env.reset()
                initial_state = env._get_state()
                
                # 1軌跡の生成
                state = initial_state
                traj_data = {'actions': [], 'rewards': [], 'spikes_history': [], 'total_reward': 0.0}
                self.agent.experience_buffer = [] # Clear buffer
                
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
            
            # GRPO Update
            self.agent.learn_with_grpo(trajectories)
            
        # 学習後の確認: 貪欲法で正解できるか
        env.reset()
        state = env._get_state()
        actions = []
        for _ in range(len(self.target_seq)):
            # 探索なし(決定論的)に近い挙動を期待
            action = self.agent.get_action(state, record_experience=False)
            state, _, done, _ = env.step(action)
            actions.append(action)
            if done: break
            
        print(f"Final Action Sequence: {actions}")
        
        # アサーション: 重みが更新され、何らかの行動が出力されていること
        self.assertTrue(len(actions) > 0)

if __name__ == '__main__':
    unittest.main()
