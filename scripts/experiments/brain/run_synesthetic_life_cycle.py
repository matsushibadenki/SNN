# ファイルパス: scripts/experiments/brain/run_synesthetic_life_cycle.py
# 日本語タイトル: Synesthetic Life Cycle Simulation (Day & Night)
# 目的: 覚醒(探索・行動)と睡眠(夢・記憶定着)のサイクルを回し、
#       単一学習エンジンが自律的に成長する過程をシミュレーションする。

from snn_research.cognitive_architecture.synesthetic_sleep import SynestheticSleepManager
from snn_research.agent.synesthetic_agent import SynestheticAgent
from snn_research.models.experimental.brain_v4 import SynestheticBrain
from snn_research.core.architecture_registry import ArchitectureRegistry
import os
import sys
import torch
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())


# ロギング設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LifeCycleSim")

# --- 簡易環境クラス (前回の拡張) ---


class DayNightEnvironment:
    def __init__(self, config):
        self.device = config['device']
        self.sensory_dims = config['sensory_configs']
        self.pos = torch.zeros(1, 2, device=self.device)
        self.target = torch.randn(1, 2, device=self.device)  # 目標地点
        self.step_count = 0

    def reset(self):
        self.pos = torch.zeros(1, 2, device=self.device)
        self.target = torch.randn(1, 2, device=self.device)
        return self._observe()

    def step(self, action: torch.Tensor):
        # 行動による移動 (action: B, 2)
        move = torch.clamp(action, -1.0, 1.0) * 0.2
        self.pos += move

        # 報酬: 目標に近いほど高い
        dist = torch.dist(self.pos, self.target)
        reward = -dist  # 距離が近いほど0に近づく（最大化）

        done = dist < 0.1
        if done:
            self.target = torch.randn(1, 2, device=self.device)  # 新しい目標

        return self._observe(), reward, done

    def _observe(self):
        # 目標との相対位置を視覚・触覚としてエンコード
        rel_pos = self.target - self.pos

        obs = {}
        # Vision: 相対位置を視覚パターン化
        if 'vision' in self.sensory_dims:
            dim = self.sensory_dims['vision']
            obs['vision'] = torch.sin(
                rel_pos[0, 0] * torch.linspace(0, 10, dim, device=self.device)).view(1, 1, dim)

        # Tactile: 壁(座標限界)に近いと反応
        if 'tactile' in self.sensory_dims:
            dim = self.sensory_dims['tactile']
            wall_dist = 2.0 - torch.abs(self.pos).max()
            contact = (wall_dist < 0.2).float()
            obs['tactile'] = (torch.ones(
                1, 1, dim, device=self.device) * contact)

        return obs


def main():
    logger.info("🌅 Starting Synesthetic Life Cycle Simulation...")

    # 1. Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        'device': device,
        'action_dim': 2,
        'vocab_size': 1000,
        'brain_d_model': 128,
        'wm_d_model': 128,
        'sensory_configs': {'vision': 64, 'tactile': 16}
    }

    # 2. Build Agent System
    logger.info("🧠 Building Brain & World Model...")

    brain = SynestheticBrain(
        vocab_size=config['vocab_size'],
        d_model=config['brain_d_model'],
        num_layers=2,
        time_steps=8,
        tactile_dim=config['sensory_configs']['tactile'],
        device=device
    )

    wm_config = {
        'd_model': config['wm_d_model'],
        'd_state': 32,
        'num_layers': 2,
        'time_steps': 8,
        'action_dim': config['action_dim'],
        'sensory_configs': config['sensory_configs'],
        'neuron': {'type': 'lif'},
        'use_bitnet': False
    }
    world_model = ArchitectureRegistry.build(
        "spiking_world_model", wm_config, 0).to(device)

    agent = SynestheticAgent(brain, world_model, config['action_dim'], device)

    # Sleep Manager
    sleep_manager = SynestheticSleepManager(agent)

    # Environment
    env = DayNightEnvironment(config)

    # 3. Simulation Loop (Days)
    num_days = 3
    steps_per_day = 20

    for day in range(1, num_days + 1):
        logger.info(f"\n=== 📅 DAY {day} START ===")

        # --- Day Phase: Activity & Experience Gathering ---
        obs = env.reset()
        daily_memories = []  # 短期記憶バッファ
        total_reward = 0.0

        # World Model学習用オプティマイザ (日中はWMを学習)
        wm_optimizer = torch.optim.AdamW(world_model.parameters(), lr=1e-3)

        for step in range(steps_per_day):
            # 行動決定
            action = agent.step(obs)

            # 環境応答
            next_obs, reward, done = env.step(action)
            total_reward += reward.item()

            # 記憶の保存 (重要な瞬間のみ保存するなどの選別も可能)
            if step % 5 == 0:  # 間引き保存
                daily_memories.append(obs)

            # World Model Online Learning (日々の学習)
            world_model.train()
            wm_optimizer.zero_grad()

            # 入力整形 (Time次元あわせ)
            # Obs: Dict[str, (1, 1, D)]
            inputs = obs
            act_in = action.view(1, 1, -1)

            _, recons, _ = world_model(inputs, act_in)

            # Loss (再構成誤差)
            wm_loss = torch.tensor(0.0, device=device)
            for k, v in recons.items():
                if k in next_obs:
                    wm_loss += torch.nn.functional.mse_loss(v, next_obs[k])

            wm_loss.backward()
            wm_optimizer.step()

            obs = next_obs

        logger.info(
            f"🌞 Day {day} Summary: Total Reward = {total_reward:.2f}, Memories Stored = {len(daily_memories)}")

        # --- Night Phase: Sleep & Consolidation ---
        if daily_memories:
            logger.info(f"=== 🌙 NIGHT {day} START ===")

            # 睡眠による記憶の定着
            # 日中に学習したWorld Modelを使って夢を見、Brainをチューニングする
            _ = sleep_manager.enter_sleep_cycle(
                initial_memories=daily_memories,
                num_cycles=3
            )

            logger.info(f"✨ Night {day} Complete. Brain Plasticity Updated.")
        else:
            logger.warning("No memories to consolidate today.")

    logger.info("\n✅ Life Cycle Simulation Completed. The agent has grown.")


if __name__ == "__main__":
    main()
