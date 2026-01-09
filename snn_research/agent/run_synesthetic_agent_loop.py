# ファイルパス: scripts/agents/run_synesthetic_agent_loop.py
# 日本語タイトル: Synesthetic Agent Runtime Loop
# 目的: 五感統合エージェント(SynestheticAgent)を擬似環境で動作させ、
#       知覚-行動ループと世界モデルによる予測(Dream)のサイクルを検証する。

from snn_research.agent.synesthetic_agent import SynestheticAgent
from snn_research.models.experimental.brain_v4 import SynestheticBrain
from snn_research.core.architecture_registry import ArchitectureRegistry
import os
import sys
import torch
import torch.nn as nn
import logging
from typing import Dict, Any

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())


# ロギング設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentLoop")


class DummyEnvironment:
    """
    五感データを提供する擬似環境クラス。
    エージェントの行動に応じて環境の状態（視覚・触覚など）を変化させる。
    """

    def __init__(self, config: Dict[str, Any]):
        self.device = config['device']
        self.sensory_dims = config['sensory_configs']
        self.step_count = 0
        self.state_pos = torch.zeros(1, 2, device=self.device)  # (x, y)

    def reset(self) -> Dict[str, torch.Tensor]:
        self.step_count = 0
        self.state_pos = torch.zeros(1, 2, device=self.device)
        return self._get_observation()

    def step(self, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        行動を受け取り、状態を更新して新しい観測を返す。
        action: (B, 2) - 移動ベクトルと仮定
        """
        self.step_count += 1

        # 状態更新 (移動)
        # actionは-1~1の範囲と想定
        self.state_pos += action * 0.1

        return self._get_observation()

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        """現在の状態に基づいた五感データを生成"""
        obs = {}
        batch_size = 1  # デモ用固定

        # 1. Vision: 位置に応じたパターン
        if 'vision' in self.sensory_dims:
            dim = self.sensory_dims['vision']
            # 位置情報をsin波で高次元化
            freq = torch.linspace(0.1, 5.0, dim, device=self.device)
            # (B, 1, D)
            obs['vision'] = torch.sin(
                self.state_pos.mean() * freq).unsqueeze(0).unsqueeze(0)

        # 2. Tactile: 特定エリアで反応
        if 'tactile' in self.sensory_dims:
            dim = self.sensory_dims['tactile']
            # 原点から離れると壁に触れると仮定
            dist = torch.norm(self.state_pos)
            contact = (dist > 1.0).float()
            obs['tactile'] = (torch.randn(batch_size, 1, dim,
                              device=self.device) * contact)

        return obs


def main():
    logger.info("🤖 Initializing Synesthetic Agent System...")

    # --- 1. Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        'device': device,
        'action_dim': 2,
        'vocab_size': 1000,
        'brain_d_model': 128,
        'wm_d_model': 128,
        'sensory_configs': {
            'vision': 64,   # デモ用に小さく設定
            'tactile': 16,
            'audio': 16,
            'olfactory': 8
        }
    }

    # --- 2. Build Components ---

    # A. Brain (思考エンジン)
    logger.info("   - Building Synesthetic Brain...")
    brain = SynestheticBrain(
        vocab_size=config['vocab_size'],
        d_model=config['brain_d_model'],
        num_layers=2,
        time_steps=8,
        tactile_dim=config['sensory_configs']['tactile'],
        olfactory_dim=config['sensory_configs']['olfactory'],
        device=device
    )

    # B. World Model (予測エンジン)
    logger.info("   - Building Spiking World Model...")
    # ArchitectureRegistryを使ってビルド
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

    # C. Agent (統合)
    logger.info("   - assembling Agent...")
    agent = SynestheticAgent(
        brain=brain,
        world_model=world_model,
        action_dim=config['action_dim'],
        device=device
    )

    # D. Environment
    env = DummyEnvironment(config)

    # --- 3. Runtime Loop ---
    logger.info("🚀 Starting Active Learning Loop...")

    # 学習用オプティマイザ (簡易的にWorldModelのみ学習させる例)
    wm_optimizer = torch.optim.AdamW(world_model.parameters(), lr=0.001)

    obs = env.reset()
    total_steps = 10

    for step in range(total_steps):
        logger.info(f"\n[Step {step+1}/{total_steps}]")

        # 1. Dream (Before Act) - 行動前のシミュレーション
        logger.info("   💭 Dreaming future possibilities...")
        _ = agent.dream(obs, horizon=5)
        logger.info("      -> Imagined 5 steps into the future.")

        # 2. Act (Brain Decision)
        # 外部からの指示 (任意)
        instruction = "Explore the environment safely."

        logger.info("   🧠 Thinking and Acting...")
        action = agent.step(obs, instruction=instruction)
        logger.info(
            f"      -> Action decided: {action[0].detach().cpu().numpy()}")

        # 3. Environment Response
        next_obs = env.step(action)

        # 4. Learn (World Model Update) - 実体験に基づく学習
        # 「予測していた結果」と「実際の結果」の誤差を学習する
        logger.info("   📚 Learning from experience (World Model Update)...")

        world_model.train()
        wm_optimizer.zero_grad()

        # 簡易学習: 1ステップ前の観測+行動 -> 現在の観測 を予測できたか？
        # (本来はReplayBufferを使うが、ここではオンライン学習の簡易実装)

        # 入力データの整形 (Time次元を追加)
        current_inputs = {k: v.unsqueeze(
            1) if v.dim() == 2 else v for k, v in obs.items()}
        # ActionもTime次元追加
        current_action_seq = action.unsqueeze(1)

        # 予測実行
        _, reconstructions, _ = world_model(current_inputs, current_action_seq)

        # 損失計算 (各感覚の再構成誤差)
        loss = torch.tensor(0.0, device=device)
        for mod, pred in reconstructions.items():
            if mod in next_obs:
                target = next_obs[mod]
                if target.dim() == 2:
                    target = target.unsqueeze(1)

                # 次元合わせ
                if pred.shape != target.shape:
                    # 簡易リサイズ (実運用では形状を厳密に管理)
                    continue

                loss += nn.MSELoss()(pred, target)

        loss.backward()
        wm_optimizer.step()
        logger.info(f"      -> World Model Loss: {loss.item():.4f}")

        # 状態更新
        obs = next_obs

        # 短い休憩 (ログを見やすくするため)
        # time.sleep(0.5)

    logger.info("\n✅ Synesthetic Agent Loop Completed Successfully.")


if __name__ == "__main__":
    main()
