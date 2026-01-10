# ファイルパス: scripts/demos/brain/run_structural_plasticity_demo.py
# 日本語タイトル: Structural Plasticity Demo (Synaptic Rewiring)
# 目的・内容:
#   ROADMAP Phase 3.2 "Self-Evolution" 実証。
#   学習に行き詰まった（あるいは定期的な睡眠）と仮定し、
#   シナプスの刈り込み(Pruning)と新生(Regrowth)を実行して、
#   脳の配線構造が変化する様子を可視化する。

import os
import sys
import torch
import torch.nn as nn
import logging
import numpy as np

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

from snn_research.evolution.structural_plasticity import StructuralPlasticity  # noqa: E402


def visualize_weights(layer, title):
    """重み行列のヒートマップを表示（CUI環境では統計情報のみログ出力）"""
    weights = layer.weight.data.cpu().numpy()
    abs_w = np.abs(weights)
    sparsity = (abs_w == 0).mean() * 100

    logger.info(f"📊 [{title}]")
    logger.info(f"   Shape: {weights.shape}")
    logger.info(f"   Sparsity (Zeroed): {sparsity:.1f}%")
    logger.info(f"   Mean Abs Weight: {abs_w.mean():.4f}")
    logger.info(f"   Max Weight: {abs_w.max():.4f}")
    return weights


def run_plasticity_demo():
    print("""
    ============================================================
       🧬 STRUCTURAL PLASTICITY DEMO (Synaptic Rewiring) 🧬
    ============================================================
    """)

    device = "cpu"

    # 1. Setup a simple network
    # 入力10 -> 隠れ20 -> 出力10 の単純なMLP
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    ).to(device)

    # 初期化（ランダム）
    logger.info("🧠 Initializing Neural Network...")
    visualize_weights(model[0], "Layer 1 (Before)")

    # 2. Initialize Evolution Engine
    # 20%のシナプスを入れ替える設定
    plasticity_engine = StructuralPlasticity(
        model,
        config={
            "pruning_rate": 0.2,
            "growth_rate": 0.2,
            "noise_std": 0.1
        }
    )

    # 3. Simulate "Learning" (making some weights important)
    logger.info("\n📚 Simulating Learning (Differentiation)...")
    # 一部の重みを意図的に大きくする（重要な接続を模倣）
    with torch.no_grad():
        # Layer 0 の最初の5ニューロンへの結合を強化
        model[0].weight.data[:5, :] *= 5.0

    visualize_weights(model[0], "Layer 1 (After Learning)")

    # 4. Trigger Structural Evolution (Sleep/Optimization)
    logger.info("\n🌙 Triggering Structural Plasticity (Rewiring)...")
    stats = plasticity_engine.evolve_structure()

    logger.info(
        f"   ✂️ Pruned: {stats['pruned']} synapses (Weak connections removed)")
    logger.info(
        f"   🌱 Grown:  {stats['grown']} synapses (New random connections created)")

    # 5. Verify Result
    logger.info("\n🔍 Verifying Structure Change...")
    w_after = visualize_weights(model[0], "Layer 1 (After Evolution)")

    # 重要な重みが残っているか確認
    # 強化した上位ニューロンの重みは大きく、Pruningされていないはず
    strong_connections_mean = np.abs(w_after[:5, :]).mean()
    weak_connections_mean = np.abs(w_after[5:, :]).mean()

    logger.info(f"   💪 Strong Connections Mean: {strong_connections_mean:.4f}")
    logger.info(f"   🍃 Weak/New Connections Mean: {weak_connections_mean:.4f}")

    if strong_connections_mean > weak_connections_mean:
        logger.info(
            "✅ SUCCESS: Important knowledge preserved while structure evolved.")
    else:
        logger.warning(
            "⚠️ WARNING: Rewiring might have damaged important knowledge.")

    logger.info("🎉 Structural Plasticity Demo Completed.")


if __name__ == "__main__":
    run_plasticity_demo()
