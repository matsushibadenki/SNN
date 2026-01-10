# ファイルパス: scripts/demos/systems/run_fast_embodied_agent.py
# 日本語タイトル: Fast Embodied Agent Demo (Powered by BitSpikeMamba)
# 目的・内容:
#   BitSpikeMambaを視覚野に搭載した高速エージェントのデモ。
#   推論レイテンシを計測し、ROADMAP Phase 2.3 "Closing the Gap" の成果を確認する。

import os
import sys
import torch
import logging
import time

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

from snn_research.systems.embodied_vlm_agent import EmbodiedVLMAgent  # noqa: E402
from snn_research.models.transformer.spiking_vlm import SpikingVLM  # noqa: E402


def run_fast_agent():
    print("""
    =======================================================
       ⚡ FAST EMBODIED AGENT (BitSpikeMamba Edition) ⚡
    =======================================================
    """)

    device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"⚙️ Running on {device.upper()}")

    vocab_size = 1000
    img_size = 32

    # 1. Config with BitSpikeMamba
    full_config = {
        "vision_config": {
            "type": "bit_spike_mamba",  # 🐍 Using Mamba
            "hidden_dim": 64,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "num_layers": 2,
            "img_size": img_size,
            "time_steps": 1,  # T=1 for ultra-low latency reaction
            "neuron": {"type": "lif"}
        },
        "language_config": {
            "d_model": 64,
            "vocab_size": vocab_size,
            "num_layers": 2,
            "time_steps": 1
        },
        "projector_config": {"projection_dim": 64},
        "sensory_inputs": {"vision": 64},
        "use_bitnet": True  # Enable BitNet quantization
    }

    motor_config = {"action_dim": 2, "hidden_dim": 32}

    # 2. Build Agent
    logger.info("🏗️ Assembling Agent...")
    vlm_model = SpikingVLM(
        vocab_size,
        vision_config=full_config["vision_config"],
        language_config=full_config["language_config"],
        projection_dim=64,
        use_bitnet=True
    )

    agent = EmbodiedVLMAgent(vlm_model, motor_config).to(device)
    agent.eval()

    # 3. Latency Test
    logger.info("⏱️ Measuring Reaction Time (Inference Latency)...")

    dummy_image = torch.randn(1, 3, img_size, img_size).to(device)
    dummy_text = torch.tensor([[101, 20, 30]], device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = agent(dummy_image, dummy_text)

    # Measure
    num_runs = 100
    latencies = []

    # Sync for timing
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            output = agent(dummy_image, dummy_text)

            # Ensure computation is done
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            end = time.time()
            latencies.append((end - start) * 1000)

    avg_latency = sum(latencies) / len(latencies)

    logger.info(f"   🚀 Average Latency: {avg_latency:.2f} ms")

    if avg_latency < 10.0:
        logger.info("   ✅ TARGET MET: Real-time capability confirmed (<10ms).")
    else:
        logger.warning("   ⚠️ TARGET MISSED: Latency too high.")

    # 4. Verification
    action = output["action_pred"]
    logger.info(f"   🤖 Action Output: {action.cpu().tolist()}")
    logger.info("🎉 Fast Embodied Agent Ready.")


if __name__ == "__main__":
    run_fast_agent()
