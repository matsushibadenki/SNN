# ファイルパス: scripts/demos/systems/run_sleep_dream_demo.py
# 日本語タイトル: Sleep & Dream Demo (Memory Consolidation) v1.1
# 目的・内容:
#   1. [Awake] エージェントが経験を積み、短期記憶に保存する。
#   2. [Sleep] 重要な経験をリプレイし、モデルを更新する。
#   3. [Wake] 記憶が定着しているか（Lossが下がっているか）を確認する。
#   [Fix] 検証時のバッチサイズを増やし、Contrastive Lossが正しく計算されるように修正。

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
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator  # noqa: E402


def run_sleep_demo():
    logger.info("🛌 Starting Sleep & Dream Consolidation Demo...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 1000
    img_size = 32

    # 1. Build Agent
    full_config = {
        "vision_config": {"type": "cnn", "hidden_dim": 64, "img_size": img_size, "time_steps": 4, "neuron": {"type": "lif"}},
        "language_config": {"d_model": 64, "vocab_size": vocab_size, "num_layers": 2, "time_steps": 4},
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

    agent = EmbodiedVLMAgent(vlm_model, motor_config).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3)

    # 2. Initialize Sleep System
    sleeper = SleepConsolidator(
        agent, optimizer, buffer_size=50, batch_size=4, device=device)

    # 3. Phase 1: Awake & Experience (Short-term Memory Accumulation)
    logger.info("☀️ [Phase 1] Awake: Exploring and accumulating memories...")

    # 重要パターン (High Reward)
    key_image = torch.randn(1, 3, img_size, img_size).to(device)
    key_text = torch.tensor([[101, 777, 777, 102]], device=device)

    # 比較用パターン (Distractors) - 検証用
    distractor_images = torch.randn(3, 3, img_size, img_size).to(device)
    distractor_texts = torch.randint(0, vocab_size, (3, 4)).to(device)

    # バッチ構築関数 (Contrastive Lossには複数サンプルが必要)
    def get_eval_batch():
        # Key Pattern + Distractors
        eval_images = torch.cat([key_image, distractor_images], dim=0)
        eval_texts = torch.cat([key_text, distractor_texts], dim=0)
        return eval_images, eval_texts

    # いくつかのランダムな経験と、少数の重要な経験を混ぜる
    for i in range(20):
        if i % 5 == 0:
            # Important experience (High Reward)
            # ノイズを加えてバリエーションを持たせる
            img = key_image + torch.randn_like(key_image) * 0.1
            txt = key_text
            reward = 1.0
            type_str = "🌟 Important"
        else:
            # Random experience (Low Reward)
            img = torch.randn(1, 3, img_size, img_size).to(device)
            txt = torch.randint(0, vocab_size, (1, 4)).to(device)
            reward = 0.1
            type_str = "sample"

        sleeper.store_experience(img, txt, reward)
        if i % 5 == 0:
            logger.info(f"   Stored {type_str} memory (Reward: {reward})")

    # 4. Check initial loss on Evaluation Batch (Before Sleep)
    agent.eval()
    with torch.no_grad():
        eval_imgs, eval_txts = get_eval_batch()
        out_before = agent.vlm(eval_imgs, eval_txts)
        loss_before = out_before["alignment_loss"].item()
    logger.info(f"📉 Loss on Key Batch BEFORE sleep: {loss_before:.4f}")

    # 5. Phase 2: Sleep & Dream (Consolidation)
    logger.info("🌙 [Phase 2] Sleeping: Replaying high-reward memories...")
    sleep_stats = sleeper.sleep(cycles=10)
    logger.info(f"   Sleep stats: {sleep_stats}")

    # 6. Phase 3: Wake & Verify (Long-term Memory Check)
    logger.info("🌅 [Phase 3] Waking up: Verifying memory retention...")
    agent.eval()
    with torch.no_grad():
        eval_imgs, eval_txts = get_eval_batch()
        out_after = agent.vlm(eval_imgs, eval_txts)
        loss_after = out_after["alignment_loss"].item()

    logger.info(f"📉 Loss on Key Batch AFTER sleep:  {loss_after:.4f}")

    improvement = loss_before - loss_after
    if improvement > 0.001:  # 微小な誤差以上の改善
        logger.info(
            f"✅ Memory Consolidated! Loss improved by {improvement:.4f}")
    else:
        logger.warning(
            f"⚠️ Memory consolidation result inconclusive (Diff: {improvement:.4f}).")

    logger.info("🎉 Sleep & Dream Demo Completed.")


if __name__ == "__main__":
    run_sleep_demo()
