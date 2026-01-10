# ファイルパス: scripts/demos/systems/run_intrinsic_motivation_demo.py
# 日本語タイトル: Intrinsic Motivation Demo (Self-Supervised Exploration)
# 目的・内容:
#   外部正解ラベルなしで、エージェントが「予測誤差」を減らすように
#   世界モデルと自身の認識機能を更新していく様子をシミュレーションする。
#   [Fix] ログ表示のための force=True 設定済み。

import os
import sys
import torch
import logging
import random

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

from snn_research.core.architecture_registry import ArchitectureRegistry  # noqa: E402
from snn_research.systems.embodied_vlm_agent import EmbodiedVLMAgent  # noqa: E402
from snn_research.systems.autonomous_learning_loop import AutonomousLearningLoop  # noqa: E402


class SimpleEnvironment:
    """
    エージェントの行動に応じて画像が変化する簡易環境。
    """

    def __init__(self, img_size=32):
        self.img_size = img_size
        # Current visual state
        self.state = torch.randn(1, 3, img_size, img_size)

    def step(self, action: torch.Tensor) -> torch.Tensor:
        """
        行動(Tensor)を受け取り、次の画像を生成して返す。
        行動の大きさに応じてノイズを加えることで「変化」を模擬する。
        """
        # Action influence
        noise = torch.randn_like(self.state) * action.mean().item() * 0.5

        # 状態遷移 (Simple drift + action)
        self.state = torch.clamp(self.state * 0.9 + noise, -1.0, 1.0)

        # Random visual glitch (Surprise element)
        if random.random() < 0.1:
            self.state = torch.randn_like(self.state)

        return self.state.clone()


def run_intrinsic_demo():
    logger.info("🧠 Starting Intrinsic Motivation Demo...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 1000
    img_size = 32

    # 1. Build Agent
    full_config = {
        "vision_config": {
            "type": "cnn", "hidden_dim": 64, "img_size": img_size, "time_steps": 4, "neuron": {"type": "lif"}
        },
        "language_config": {
            "d_model": 64, "vocab_size": vocab_size, "num_layers": 2, "time_steps": 4
        },
        "projector_config": {"projection_dim": 64},
        "sensory_inputs": {"vision": 64},
        "use_bitnet": False
    }
    motor_config = {"action_dim": 2, "hidden_dim": 32}

    try:
        vlm_model = ArchitectureRegistry.build(
            "spiking_vlm", full_config, vocab_size)
    except Exception:
        from snn_research.models.transformer.spiking_vlm import SpikingVLM
        vlm_model = SpikingVLM(
            vocab_size, full_config["vision_config"], full_config["language_config"], projection_dim=64)

    agent = EmbodiedVLMAgent(vlm_model, motor_config)

    # 2. Setup Loop
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3)
    loop = AutonomousLearningLoop(agent, optimizer, device=device)

    env = SimpleEnvironment(img_size=img_size)

    # 3. Run Self-Supervised Loop
    logger.info("🔄 Running autonomous cycles (Curiosity-driven)...")

    current_image = env.state.to(device)
    dummy_text = torch.randint(0, vocab_size, (1, 8)).to(
        device)  # Context/Goal (Fixed for now)

    steps = 10
    for step in range(steps):
        # Environment Step happens conceptually inside loop step or here
        # Here we simulate: Agent Acts -> Env Changes -> Agent Observes Next

        # 1. Peek at agent's action to step environment (Simulation hack)
        # In real loop, agent.step() would return action, then we step env.
        # But AutonomousLearningLoop.step is monolithic for training.
        # We will split the logic:
        # Get action -> Step Env -> Train

        agent.eval()
        with torch.no_grad():
            agent_out = agent(current_image, dummy_text)
            action = agent_out["action_pred"]

        # 2. Environment Impact
        next_image = env.step(action).to(device)

        # 3. Learn (Self-Supervised)
        metrics = loop.step(current_image, dummy_text, next_image)

        logger.info(
            f"   Step {step}: Loss={metrics['loss']:.4f}, PredErr={metrics['prediction_error']:.4f}, Reward={metrics['intrinsic_reward']:.4f}")

        current_image = next_image

    logger.info("✅ Autonomous loop completed without external labels.")
    logger.info("🎉 Intrinsic Motivation Demo Completed.")


if __name__ == "__main__":
    run_intrinsic_demo()
