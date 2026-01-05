# ファイルパス: snn_research/training/bio_trainer.py
# Title: Bio RL Trainer
# 修正: Agentのメソッド呼び出し型エラーなしを確認 (ReinforcementLearnerAgentにlearnを追加済み)

import torch
from tqdm import tqdm
from typing import Dict, List, Any

from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.rl_env.grid_world import GridWorldEnv

class BioRLTrainer:
    """生物学的強化学習エージェントのためのトレーナー。"""
    def __init__(self, agent: ReinforcementLearnerAgent, env: GridWorldEnv):
        self.agent = agent
        self.env = env

    def train(self, num_episodes: int) -> Dict[str, Any]:
        """強化学習の学習ループを実行する。"""
        progress_bar = tqdm(range(num_episodes))
        total_rewards: List[float] = []

        for episode in progress_bar:
            state: torch.Tensor = self.env.reset()
            done: bool = False
            episode_reward: float = 0.0
            
            while not done:
                action: int = self.agent.get_action(state)
                next_state: torch.Tensor
                reward: float
                next_state, reward, done = self.env.step(action)
                
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
            "rewards_history": total_rewards
        }