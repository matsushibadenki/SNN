# ファイルパス: scripts/demos/brain/run_recursive_evolution_demo.py
# 日本語タイトル: Recursive Evolution Demo (The AI Scientist)
# 目的・内容:
#   RecursiveImprover を使用して、モデルの構造を自動進化させるデモ。
#   評価関数としてダミーのタスク（計算コストが低く、精度が高いほど良い）を定義し、
#   世代を経るごとに「効率的で賢い設定」が発見される様子を観察する。

import os
import sys
import logging
import time
import random

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

from snn_research.evolution.recursive_improver import RecursiveImprover  # noqa: E402

# --- Dummy Evaluator ---


def mock_brain_evaluator(config: dict) -> float:
    """
    モデル設定を評価するダミー関数。
    本来は実際にモデルを構築して学習・テストするが、
    デモ時間を短縮するため、計算式でスコアを算出する。

    Target:
    - hidden_dim: 大きいほど精度が高いが、大きすぎるとペナルティ（計算コスト）
    - num_layers: 多いほど精度が高いが、深すぎると学習困難
    - time_steps: 適度が良い
    """
    h_dim = config.get("hidden_dim", 64)
    layers = config.get("num_layers", 2)
    t_steps = config.get("time_steps", 4)

    # 理想値の設定 (Target Brain Structure)
    ideal_h = 128
    ideal_l = 4

    # Accuracy Simulation (Parabola peak at ideal values)
    acc_h = 1.0 - (abs(h_dim - ideal_h) / 500.0)
    acc_l = 1.0 - (abs(layers - ideal_l) / 10.0)

    base_accuracy = (acc_h * 0.6 + acc_l * 0.4)

    # Efficiency Penalty (Larger is slower)
    cost = (h_dim * layers * t_steps) / 10000.0

    # Fitness = Accuracy - Cost (Balance)
    fitness = base_accuracy - (cost * 0.1)

    # Add random noise (Measurement noise)
    fitness += random.uniform(-0.01, 0.01)

    return max(0.0, fitness)


def run_evolution_demo():
    print("""
    ============================================================
       🧬 RECURSIVE EVOLUTION DEMO (Architecture Search) 🧬
    ============================================================
    """)

    # 1. Define Initial "Seed" Config (Weak Brain)
    base_config = {
        "hidden_dim": 32,   # Too small
        "num_layers": 1,    # Too shallow
        "time_steps": 2,
        "neuron": {
            "type": "lif",
            "base_threshold": 1.0
        }
    }

    logger.info("🌱 Initial Genome (Seed):")
    logger.info(f"   Hidden Dim: {base_config['hidden_dim']}")
    logger.info(f"   Num Layers: {base_config['num_layers']}")

    # 2. Initialize Evolution Engine
    evolver = RecursiveImprover(
        base_config=base_config,
        evaluator_func=mock_brain_evaluator,
        population_size=10,  # 1世代あたり10個体生成
        mutation_rate=0.5    # 変異確率高め
    )

    # 3. Run Evolution Loop
    generations = 10
    logger.info(
        f"\n🚀 Starting Evolution Process ({generations} generations)...")

    start_time = time.time()
    best_genome = evolver.evolve(generations=generations)
    duration = time.time() - start_time

    # 4. Result
    print("\n" + "="*40)
    logger.info("✨ Evolution Complete!")
    logger.info(f"   Total Time: {duration:.2f}s")
    logger.info("   Best Evolved Configuration:")
    logger.info(f"     Fitness:    {best_genome.fitness:.4f}")
    logger.info(
        f"     Hidden Dim: {best_genome.config['hidden_dim']} (Started at 32)")
    logger.info(
        f"     Num Layers: {best_genome.config['num_layers']} (Started at 1)")
    logger.info(f"     Time Steps: {best_genome.config['time_steps']}")
    print("="*40)

    # Analysis
    if best_genome.config['hidden_dim'] > 32 and best_genome.fitness > 0.5:
        logger.info(
            "✅ SUCCESS: The brain autonomously evolved a more complex and efficient structure.")
    else:
        logger.warning(
            "⚠️ RESULT: Evolution didn't significantly improve the structure.")

    logger.info("🎉 Recursive Evolution Demo Completed.")


if __name__ == "__main__":
    run_evolution_demo()
