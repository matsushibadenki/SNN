# ファイルパス: scripts/demos/systems/run_embodied_interaction_demo.py
# 日本語タイトル: Embodied Interaction Demo (Vision-Language-Motor)
# 目的・内容:
#   EmbodiedVLMAgentの動作デモ。
#   ダミーの画像入力に対して、テキスト応答（例: "Stop sign detected"）と
#   モーター指令（例: ブレーキ操作）が同時に生成されることを確認する。
#   [Fix] ログ表示のための force=True 設定済み。

import os
import sys
import torch
import logging

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


def run_embodied_demo():
    logger.info("🦾 Starting Embodied VLM Interaction Demo...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 1000
    img_size = 32
    time_steps = 4

    # 1. Setup Configs
    full_config = {
        "vision_config": {
            "type": "cnn",
            "hidden_dim": 128,
            "img_size": img_size,
            "time_steps": time_steps,
            "neuron": {"type": "lif"}
        },
        "language_config": {  # Registry expects this key
            "d_model": 128,
            "vocab_size": vocab_size,
            "num_layers": 2,
            "time_steps": time_steps
        },
        "projector_config": {"projection_dim": 128},
        "sensory_inputs": {"vision": 128},  # Helper for builder
        "use_bitnet": False
    }

    motor_config = {
        "action_dim": 2,  # Example: [Steering, Throttle]
        "action_type": "continuous",
        "hidden_dim": 64
    }

    # 2. Build Core VLM
    logger.info("🏗️ Building SpikingVLM...")
    try:
        vlm_model = ArchitectureRegistry.build(
            "spiking_vlm", full_config, vocab_size)
    except Exception as e:
        logger.warning(
            f"Registry build failed ({e}), falling back to direct init.")
        from snn_research.models.transformer.spiking_vlm import SpikingVLM
        vlm_model = SpikingVLM(
            vocab_size=vocab_size,
            vision_config=full_config["vision_config"],
            text_config=full_config["language_config"],
            projection_dim=128
        )

    vlm_model = vlm_model.to(device)

    # 3. Build Embodied Agent (System Integration)
    agent = EmbodiedVLMAgent(vlm_model, motor_config).to(device)
    logger.info("🤖 Embodied Agent successfully assembled.")

    # 4. Interaction Simulation
    logger.info("🧪 Simulating Interaction Loop...")

    # Scenario: Agent sees an image and receives a prompt "What should you do?"
    dummy_image = torch.randn(1, 3, img_size, img_size).to(device)
    # [CLS, "What", "do", "?"]
    dummy_prompt = torch.tensor([[101, 20, 30, 40]], device=device)

    # Run Inference
    output = agent.act_and_speak(dummy_image, dummy_prompt)

    generated_text = output["generated_tokens"].cpu().tolist()[0]
    action_values = output["action"].cpu().tolist()[0]

    logger.info("--- Simulation Results ---")
    logger.info(f"👀 Visual Input: Tensor shape {dummy_image.shape}")
    logger.info(f"🗣️  Generated Response (Tokens): {generated_text}")
    logger.info(f"🦾 Motor Action (Steering, Throttle): {action_values}")

    # Verify outputs
    if len(generated_text) > 0 and len(action_values) == motor_config["action_dim"]:
        logger.info(
            "✅ Multimodal Integration (Vision -> Language + Motor) Verified!")
    else:
        logger.error("❌ Output format verification failed.")

    logger.info("🎉 Embodied Demo Completed.")


if __name__ == "__main__":
    run_embodied_demo()
