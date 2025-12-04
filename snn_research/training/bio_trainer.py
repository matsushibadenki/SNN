# ファイルパス: snn_research/training/bio_trainer.py
# Title: 生物学的強化学習用トレーナー
# Description: 強化学習のパラダイムに合わせ、エージェントと環境を引数に取るように変更。
#              エピソードベースの学習ループ（行動選択 -> 環境作用 -> 学習）を実装。
# 修正点:
# - mypyエラーを解消するため、GridWorldEnvに対応。
# - 複数ステップからなるエピソードベースの学習ループに修正。
# - mypyエラー `Incompatible types in assignment` を解消するため、
#   `episode_reward` をfloatで初期化するように修正。
#
# 修正 (v2):
# - 健全性チェック (health-check) のために、報酬履歴を返すように修正。

import torch
from tqdm import tqdm  # type: ignore
# --- ▼ 修正 (v2): List, Dict をインポート ▼ ---
from typing import Dict, List, Any # Any をインポート
# --- ▲ 修正 (v2) ▲ ---

from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.rl_env.grid_world import GridWorldEnv

class BioRLTrainer:
    """生物学的強化学習エージェントのためのトレーナー。"""
    def __init__(self, agent: ReinforcementLearnerAgent, env: GridWorldEnv):
        self.agent = agent
        self.env = env

    def train(self, num_episodes: int) -> Dict[str, Any]: # Dict[str, float] を Dict[str, Any] に変更
        """強化学習の学習ループを実行する。"""
        progress_bar = tqdm(range(num_episodes))
        total_rewards: List[float] = [] # 報酬履歴をリストで保持

        for episode in progress_bar:
            state: torch.Tensor = self.env.reset()
            done: bool = False
            episode_reward: float = 0.0 # floatで初期化
            
            while not done:
                # 1. 行動選択
                action: int = self.agent.get_action(state)
                
                # 2. 環境との相互作用
                next_state: torch.Tensor
                reward: float
                next_state, reward, done = self.env.step(action)
                
                # 3. 学習
                self.agent.learn(reward)
                
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)
            avg_reward: float = sum(total_rewards[-20:]) / len(total_rewards[-20:])
            
            progress_bar.set_description(f"Bio RL Training Episode {episode+1}/{num_episodes}")
            progress_bar.set_postfix({"Last Reward": f"{episode_reward:.2f}", "Avg Reward (last 20)": f"{avg_reward:.3f}"})

        final_avg_reward: float = sum(total_rewards) / num_episodes if num_episodes > 0 else 0.0
        print(f"Training finished. Final average reward: {final_avg_reward:.4f}")
        
        return {
            "final_average_reward": final_avg_reward,
            "rewards_history": total_rewards # health-checkでのプロット用
        }