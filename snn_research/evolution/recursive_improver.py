# ファイルパス: snn_research/evolution/recursive_improver.py
# 日本語タイトル: Recursive Improver (Evolutionary Engine) v1.0
# 目的・内容:
#   ROADMAP Phase 3.3 "Self-Evolution" 対応。
#   遺伝的アルゴリズム(GA)を用いて、モデルのアーキテクチャ設定(Config)を進化させる。
#   「精度(Accuracy)」と「効率(Efficiency)」のバランスが良い個体を選抜する。

import copy
import random
import logging
from typing import Dict, Any, List, Callable

logger = logging.getLogger(__name__)


class Genome:
    """1つのモデル設定（個体）を表すクラス"""

    def __init__(self, config: Dict[str, Any], fitness: float = 0.0):
        self.config = copy.deepcopy(config)
        self.fitness = fitness
        self.generation = 0


class RecursiveImprover:
    """
    自己改善エンジン。
    モデルの構成（ニューロン数、層数、パラメータなど）を変異させ、
    より良い性能を持つ構造を探索する。
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        evaluator_func: Callable[[Dict[str, Any]], float],
        population_size: int = 5,
        mutation_rate: float = 0.3
    ):
        self.base_config = base_config
        self.evaluator = evaluator_func  # Configを受け取りFitnessを返す関数
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation_count = 0

        # 初期個体群の生成
        self.population: List[Genome] = []
        logger.info(
            f"🧬 Recursive Improver initialized. PopSize: {population_size}")

    def _mutate_value(self, value: Any, key: str) -> Any:
        """値をランダムに変異させる"""
        if isinstance(value, int):
            # 整数パラメータ (例: hidden_dim, layers)
            if "dim" in key or "width" in key:
                # 2の倍数で増減
                change = random.choice([-16, -8, 8, 16])
                return max(16, value + change)
            elif "layer" in key or "depth" in key:
                # 層数の増減
                change = random.choice([-1, 1])
                return max(1, value + change)
            else:
                # その他 (Time stepsなど)
                change = random.choice([-1, 0, 1])
                return max(1, value + change)

        elif isinstance(value, float):
            # 実数パラメータ (例: learning_rate, threshold)
            change = random.uniform(0.8, 1.2)
            return value * change

        elif isinstance(value, bool):
            # フラグ反転 (低確率)
            return not value if random.random() < 0.1 else value

        return value

    def _mutate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """設定辞書を再帰的に走査して変異させる"""
        new_config = copy.deepcopy(config)

        for k, v in new_config.items():
            if isinstance(v, dict):
                new_config[k] = self._mutate_config(v)
            else:
                # 変異確率に基づく変更
                if random.random() < self.mutation_rate:
                    # 特定のキーだけ変異対象にする（簡易化）
                    if k in ["hidden_dim", "num_layers", "d_model", "time_steps", "base_threshold"]:
                        # original = v
                        new_config[k] = self._mutate_value(v, k)
                        # logger.debug(f"   Mutation: {k} {original} -> {new_config[k]}")

        return new_config

    def evolve(self, generations: int = 1) -> Genome:
        """
        指定世代数だけ進化を実行する。
        """
        # 初回のみベース個体を評価
        if not self.population:
            logger.info("🌱 Evaluating Adam (Base Individual)...")
            base_fitness = self.evaluator(self.base_config)
            self.population = [Genome(self.base_config, base_fitness)]

        for gen in range(generations):
            self.generation_count += 1
            logger.info(
                f"🔄 Generation {self.generation_count} started. Best Fitness: {self.population[0].fitness:.4f}")

            # 1. Selection (Elitism)
            # 現在のベスト個体を親とする
            parent = self.population[0]

            # 2. Reproduction & Mutation
            offsprings = []
            for i in range(self.population_size - 1):  # 親以外の子を作成
                mutated_conf = self._mutate_config(parent.config)
                child = Genome(mutated_conf)
                child.generation = self.generation_count
                offsprings.append(child)

            # 3. Evaluation
            # 並列化可能だが、ここでは直列実行
            for i, child in enumerate(offsprings):
                # 評価関数を実行（実際にモデルを作ってテスト）
                try:
                    score = self.evaluator(child.config)
                    child.fitness = score
                    # logger.info(f"   Child {i+1}: Fitness = {score:.4f}")
                except Exception as e:
                    logger.warning(
                        f"   Child {i+1} died (Invalid Config): {e}")
                    child.fitness = -1.0

            # 4. Survival of the Fittest
            # 親 + 子の中でランキング
            pool = [parent] + offsprings
            pool.sort(key=lambda x: x.fitness, reverse=True)

            # 上位1個体のみ残す（今回はSimple Hill Climbingに近いGA）
            # または多様性維持のため上位N個を残す
            # Keep top N for next parenthood if needed
            self.population = pool[:self.population_size]

            best = self.population[0]
            logger.info(
                f"🏆 Gen {self.generation_count} Winner: Fitness {best.fitness:.4f} (Dims: {best.config.get('hidden_dim', 'N/A')}, Layers: {best.config.get('num_layers', 'N/A')})")

        return self.population[0]
