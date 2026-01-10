# ファイルパス: scripts/runners/run_autonomous_life_cycle.py
# 日本語タイトル: Autonomous Life Cycle Runner
# 目的・内容:
#   AutonomousLearningLoopを用いて、エージェントの「一生（ライフサイクル）」を実行する。
#   ダミーの視覚・言語入力を用いて、覚醒と睡眠のサイクルが正しく回ることを検証する。

from snn_research.systems.embodied_vlm_agent import EmbodiedVLMAgent
from snn_research.systems.autonomous_learning_loop import AutonomousLearningLoop
import os
import sys
import torch
import torch.optim as optim
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


# ログ設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("🚀 Starting Autonomous Life Cycle Simulation...")

    # 1. Setup Environment & Agent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 仮のエージェント設定
    agent_config = {
        "vision_dim": 3,     # RGB
        "text_dim": 768,     # Embedding size
        "hidden_dim": 512,
        "action_dim": 64
    }

    # エージェントの初期化 (実在クラスまたはモックが必要)
    # ここでは既存のEmbodiedVLMAgentを使用
    try:
        agent = EmbodiedVLMAgent(**agent_config).to(device)
    except Exception as e:
        logger.warning(
            f"Failed to init real agent: {e}. Using Mock for demonstration.")
        agent = MockAgent(agent_config).to(device)

    # オプティマイザ
    optimizer = optim.AdamW(agent.parameters(), lr=1e-4)

    # 2. Initialize Autonomous Loop
    # 疲労閾値を低く設定して、すぐに睡眠サイクルが見られるようにする
    life_cycle = AutonomousLearningLoop(
        agent=agent,
        optimizer=optimizer,
        device=device,
        energy_capacity=100.0,
        fatigue_threshold=20.0
    )

    # 3. Simulation Loop (Days)
    num_steps = 100

    logger.info(f"⏳ Running simulation for {num_steps} steps...")

    for step in range(num_steps):
        # Mock Sensory Input (環境からの入力)
        # 本来はカメラやシミュレータから取得
        current_image = torch.randn(1, 3, 64, 64).to(device)
        current_text = torch.randn(1, 10, 768).to(device)  # Text embeddings
        next_image = torch.randn(1, 3, 64, 64).to(device)  # 次の瞬間の画像

        # Step Execution
        status = life_cycle.step(current_image, current_text, next_image)

        mode = status["mode"]

        if mode == "wake":
            logger.info(
                f"Step {step:03d} [Wake]: Surprise={status['surprise']:.4f}, "
                f"Reward={status['intrinsic_reward']:.4f}, "
                f"Fatigue={status['fatigue']:.1f}/{life_cycle.fatigue_threshold}"
            )
        elif mode == "sleep":
            logger.info(
                f"Step {step:03d} [SLEEP]: 💤 Memory Consolidation Loss={status['sleep_loss']:.4f}")

    logger.info("✅ Simulation Complete.")

# --- Mock Classes for Independent Execution ---


class MockAgent(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fusion_dim = config["hidden_dim"]
        self.action_dim = config["action_dim"]
        self.mock_layer = torch.nn.Linear(
            config["vision_dim"], self.fusion_dim)

        # Dummy VLM sub-module interface
        self.vlm = self._vlm_mock

    def forward(self, img, txt):
        B = img.shape[0]
        return {
            "fused_context": torch.randn(B, self.fusion_dim, device=img.device),
            "action_pred": torch.randn(B, self.action_dim, device=img.device),
            "alignment_loss": torch.tensor(0.1, device=img.device, requires_grad=True)
        }

    def _vlm_mock(self, img, txt):
        B = img.shape[0]
        return {
            "fused_representation": torch.randn(B, self.fusion_dim, device=img.device)
        }


if __name__ == "__main__":
    main()
