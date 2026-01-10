# ファイルパス: scripts/demos/brain/run_qualia_demo.py
# 日本語タイトル: Qualia & Subjectivity Demo
# 目的・内容:
#   「同じものを見ても、気分によって感じ方が変わる」という主観性（Qualia）を実証する。
#   1. 中立な状態で画像を見る。
#   2. 恐怖状態で同じ画像を見る。
#   3. 喜び状態で同じ画像を見る。
#   それぞれの状態で生成された「クオリア」の距離を測定し、内部体験の変化を確認する。

from snn_research.cognitive_architecture.qualia_synthesizer import QualiaSynthesizer
import os
import sys
import torch
import logging

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


def run_qualia_demo():
    print("""
    ============================================================
       🌈 QUALIA DEMO (Subjective Experience Synthesis) 🌈
    ============================================================
    """)

    device = "cpu"
    sensory_dim = 64
    emotion_dim = 4  # [Fear, Anger, Joy, Sadness]

    # 1. Initialize Synthesizer
    qualia_engine = QualiaSynthesizer(sensory_dim, emotion_dim).to(device)

    # 2. Prepare Inputs
    # Stimulus: "A Forest" (Fixed Vector)
    forest_stimulus = torch.randn(1, sensory_dim).to(device)
    # Normalize for stability
    forest_stimulus = forest_stimulus / forest_stimulus.norm()

    logger.info("🌲 Stimulus: 'A Forest' (Objective Data)")

    # 3. Define Internal States (Emotions)
    states = {
        "Neutral": torch.tensor([[0.1, 0.1, 0.1, 0.1]]),  # Flat
        "Fear 😨": torch.tensor([[0.9, 0.1, 0.0, 0.1]]),  # High Fear
        "Joy 😄":  torch.tensor([[0.0, 0.0, 0.9, 0.0]]),  # High Joy
    }

    qualia_memory = {}

    # 4. Generate Qualia for each state
    logger.info("\n🧪 Generating Subjective Experiences...")

    for name, emotion in states.items():
        emotion = emotion.to(device)

        # Forward pass
        output = qualia_engine(forest_stimulus, emotion)
        quale = output["qualia"]
        mod = output["modulation"]

        qualia_memory[name] = quale

        # Visualize Modulation (How emotion filtered the input)
        # 平均的なフィルタ強度を表示
        filter_strength = mod.mean().item()
        logger.info(
            f"   [{name}] processing 'Forest' -> Filter Intensity: {filter_strength:.2f}")

    # 5. Measure Subjective Distances
    logger.info("\n📏 Measuring Phenomenological Distances (Cosine Distance)...")

    # Neutral vs Fear
    dist_fear = qualia_engine.compute_subjective_distance(
        qualia_memory["Neutral"], qualia_memory["Fear 😨"]
    )

    # Neutral vs Joy
    dist_joy = qualia_engine.compute_subjective_distance(
        qualia_memory["Neutral"], qualia_memory["Joy 😄"]
    )

    # Fear vs Joy
    dist_contrast = qualia_engine.compute_subjective_distance(
        qualia_memory["Fear 😨"], qualia_memory["Joy 😄"]
    )

    logger.info(f"   Neutral <-> Fear: {dist_fear:.4f} (Dark/Scary Forest)")
    logger.info(f"   Neutral <-> Joy : {dist_joy:.4f} (Bright/Happy Forest)")
    logger.info(
        f"   Fear    <-> Joy : {dist_contrast:.4f} (Completely different worlds)")

    # 6. Conclusion
    if dist_contrast > 0.1:
        logger.info(
            "\n✅ SUCCESS: The agent experienced the 'Forest' differently based on internal state.")
        logger.info(
            "   Objectively, the data was identical. Subjectively, the Qualia transformed.")
    else:
        logger.warning(
            "\n⚠️ WARNING: Qualia differentiation failed. Check weights.")

    logger.info("🎉 Qualia Demo Completed.")


if __name__ == "__main__":
    run_qualia_demo()
