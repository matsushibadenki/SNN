# ファイルパス: snn_research/evolution/recursive_improver.py
# 日本語タイトル: Recursive Improver (Fixed Imports)
# 目的: typingモジュールのインポート不足によるNameErrorを修正。

import copy
import random
import logging
from typing import Dict, Any, List, Callable, Optional, Tuple

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
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        evaluator_func: Optional[Callable[[Dict[str, Any]], float]] = None,
        population_size: int = 5,
        mutation_rate: float = 0.3
    ):
        self.base_config = base_config
        self.evaluator = evaluator_func
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation_count = 0

        # 初期個体群
        self.population: List[Genome] = [Genome(base_config)]

        logger.info(
            f"🧬 Recursive Improver initialized. PopSize: {population_size}")

    def _mutate_value(self, value: Any, key: str) -> Any:
        """値をランダムに変異させる"""
        if isinstance(value, int):
            # 整数パラメータ
            if "dim" in key or "width" in key:
                change = int(random.choice([-16, -8, 8, 16]))
                return max(16, value + change)
            elif "layer" in key or "depth" in key:
                change = int(random.choice([-1, 1]))
                return max(1, value + change)
            else:
                change = int(random.choice([-1, 0, 1]))
                return max(1, value + change)

        elif isinstance(value, float):
            # 実数パラメータ
            factor = random.uniform(0.8, 1.2)
            return value * factor

        elif isinstance(value, bool):
            return not value if random.random() < 0.1 else value

        return value

    def _mutate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """設定辞書を再帰的に走査して変異させる"""
        new_config = copy.deepcopy(config)
        for k, v in new_config.items():
            if isinstance(v, dict):
                new_config[k] = self._mutate_config(v)
            else:
                if random.random() < self.mutation_rate:
                    if k in ["hidden_dim", "num_layers", "d_model", "time_steps", "base_threshold"]:
                        new_config[k] = self._mutate_value(v, k)
        return new_config

    def spawn_generation(self, pop_size: int = 2) -> List[Any]:
        """次世代の候補を生成"""
        if not self.population:
            # Fallback if population is empty
            self.population = [Genome(self.base_config)]

        parent = self.population[0]
        candidates = []

        for _ in range(pop_size):
            mutated_conf = self._mutate_config(parent.config)
            child = Genome(mutated_conf)
            child.generation = self.generation_count + 1
            candidates.append(child)

        return candidates

    def evaluate_and_select(
        self,
        candidates: List[Any],
        eval_func: Callable[[Any], float]
    ) -> Tuple[Any, float]:
        """候補評価と選択"""
        best_candidate = None
        best_score = -999.0

        evaluated_genomes = []

        for cand in candidates:
            score = eval_func(cand)

            if score > best_score:
                best_score = score
                best_candidate = cand

            if isinstance(cand, Genome):
                cand.fitness = score
                evaluated_genomes.append(cand)

        if evaluated_genomes:
            pool = self.population + evaluated_genomes
            pool.sort(key=lambda x: x.fitness, reverse=True)
            self.population = pool[:self.population_size]

        self.generation_count += 1
        return best_candidate, best_score

    def evolve(self, generations: int = 1) -> Genome:
        """スタンドアロン進化実行"""
        if not self.evaluator:
            raise ValueError(
                "Evaluator function required for standalone evolution.")

        if self.population[0].fitness == 0.0:
            self.population[0].fitness = self.evaluator(
                self.population[0].config)

        for _ in range(generations):
            candidates = self.spawn_generation(self.population_size - 1)
            for cand in candidates:
                cand.fitness = self.evaluator(cand.config)

            pool = self.population + candidates
            pool.sort(key=lambda x: x.fitness, reverse=True)
            self.population = pool[:self.population_size]

            best = self.population[0]
            logger.info(
                f"Gen {self.generation_count}: Best Fitness {best.fitness:.4f}")
            self.generation_count += 1

        return self.population[0]
