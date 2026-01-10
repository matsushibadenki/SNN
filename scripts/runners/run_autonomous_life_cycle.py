# ファイルパス: scripts/runners/run_autonomous_life_cycle.py
# 日本語タイトル: Autonomous Life Cycle Runner
# 目的・内容:
#   AutonomousLearningLoopを用いて、エージェントの「一生（ライフサイクル）」を実行する。
#   ダミーの視覚・言語入力を用いて、覚醒と睡眠のサイクルが正しく回ることを検証する。
#   [Fix] ログ設定の強制(force=True)と、main関数呼び出しの確実化、VLM初期化順序の修正。

# インポート前にプリントして動作確認
import logging
import torch.optim as optim
import torch
import sys
import os
from snn_research.systems.autonomous_learning_loop import AutonomousLearningLoop
from snn_research.systems.embodied_vlm_agent import EmbodiedVLMAgent
from snn_research.models.transformer.spiking_vlm import SpikingVLM
print("--- [DEBUG] Script loading... ---")


# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# ログ設定 (force=Trueで既存の設定を上書きし、確実に表示させる)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def main():
    print("--- [DEBUG] Entering main() ---")
    logger.info("🚀 Starting Autonomous Life Cycle Simulation...")

    # 1. Setup Environment & Agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ダミーの語彙サイズ
    vocab_size = 1000

    # エージェントの設定 (VLMとMotor)
    # SpikingVLMの設定
    vision_config = {
        "type": "cnn",
        "hidden_dim": 512,
        "img_size": 64,  # ダミー画像のサイズに合わせる
        "time_steps": 16
    }
    text_config = {
        "d_model": 512,
        "num_layers": 2
    }

    # MotorDecoderの設定
    motor_config = {
        "action_dim": 64,
        "hidden_dim": 512,
        "action_type": "continuous"
    }

    try:
        # まずVLMを構築
        logger.info("Building SpikingVLM...")
        vlm_model = SpikingVLM(
            vocab_size=vocab_size,
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=512
        ).to(device)

        # EmbodiedVLMAgentを初期化 (VLMインスタンスを渡す)
        logger.info("Building EmbodiedVLMAgent...")
        agent = EmbodiedVLMAgent(
            vlm_model=vlm_model,
            motor_config=motor_config
        ).to(device)
        logger.info("✅ Agent built successfully.")

    except Exception as e:
        logger.error(f"Failed to init real agent: {e}", exc_info=True)
        logger.warning("Using Mock for demonstration.")
        # モック使用時は構成辞書をフラットな形式に変換して渡す
        mock_config = {
            "vision_dim": 3,
            "text_dim": 512,
            "hidden_dim": 512,
            "action_dim": 64
        }
        agent = MockAgent(mock_config).to(device)

    # オプティマイザ
    optimizer = optim.AdamW(agent.parameters(), lr=1e-4)

    # 2. Initialize Autonomous Loop
    # 疲労閾値を低く設定して、すぐに睡眠サイクルが見られるようにする
    logger.info("Initializing AutonomousLearningLoop...")
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
        # 画像入力 [Batch, Channels, Height, Width]
        current_image = torch.randn(1, 3, 64, 64).to(device)

        # テキスト入力 [Batch, Seq_Len] (SpikingVLMはトークンIDを期待する)
        current_text = torch.randint(0, vocab_size, (1, 10)).to(device)

        # 次の瞬間の画像
        next_image = torch.randn(1, 3, 64, 64).to(device)

        # Step Execution
        # AutonomousLearningLoop内で agent(image, text) が呼ばれる
        status = life_cycle.step(current_image, current_text, next_image)

        mode = status["mode"]

        if mode == "wake":
            logger.info(
                f"Step {step:03d} [Wake]: Surprise={status.get('surprise', 0.0):.4f}, "
                f"Reward={status.get('intrinsic_reward', 0.0):.4f}, "
                f"Fatigue={status.get('fatigue', 0.0):.1f}/{life_cycle.fatigue_threshold}"
            )
        elif mode == "sleep":
            logger.info(
                f"Step {step:03d} [SLEEP]: 💤 Memory Consolidation Loss={status.get('sleep_loss', 0.0):.4f}")

    logger.info("✅ Simulation Complete.")

# --- Mock Classes for Independent Execution ---


class MockAgent(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fusion_dim = config["hidden_dim"]
        self.action_dim = config["action_dim"]
        # 入力次元などは無視して、内部で適当なサイズの層を持つ
        self.mock_layer = torch.nn.Linear(10, self.fusion_dim)

        # Dummy VLM sub-module interface
        self.vlm = self._vlm_mock

        # Mock用にProjectorもどきを持たせておく
        self.vlm.projector = type(
            'obj', (object,), {'embed_dim': self.fusion_dim})

    def forward(self, img, txt):
        B = img.shape[0]
        return {
            "fused_context": torch.randn(B, self.fusion_dim, device=img.device),
            "action_pred": torch.randn(B, self.action_dim, device=img.device),
            "alignment_loss": torch.tensor(0.1, device=img.device, requires_grad=True),
            # Dummy logits
            "logits": torch.randn(B, 10, 1000, device=img.device)
        }

    def _vlm_mock(self, img, txt):
        B = img.shape[0]
        return {
            "fused_representation": torch.randn(B, self.fusion_dim, device=img.device)
        }


if __name__ == "__main__":
    main()
