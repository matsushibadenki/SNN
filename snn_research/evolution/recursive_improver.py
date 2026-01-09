# ファイルパス: snn_research/evolution/recursive_improver.py
# Title: Recursive Self-Improver (Verbose Fix)
# Description:
# - エラーハンドリングとログ出力を強化。

import torch
import torch.nn as nn
import logging
import copy
import random
import numpy as np
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class RecursiveImprover:
    def __init__(self, target_brain: nn.Module, mutation_rate: float = 0.05):
        self.current_best_brain = target_brain
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.improvement_history: List[float] = []
        logger.info("🧬 Recursive Self-Improver initialized.")

    def spawn_generation(self, pop_size: int = 3) -> List[nn.Module]:
        offspring = []
        offspring.append(self.current_best_brain) # エリート
        
        for i in range(pop_size - 1):
            try:
                # Deepcopy can be slow
                child = copy.deepcopy(self.current_best_brain)
                self._mutate(child)
                offspring.append(child)
            except Exception as e:
                logger.error(f"Failed to spawn child: {e}")
                # フォールバック: 元の脳を返す
                offspring.append(self.current_best_brain)
            
        return offspring

    def _mutate(self, brain: nn.Module):
        with torch.no_grad():
            params = list(brain.parameters())
            # 全パラメータではなくランダムにいくつか選んで変異させる（高速化）
            target_params = random.sample(params, k=min(len(params), 5))
            
            for param in target_params:
                if param.requires_grad:
                    noise = torch.randn_like(param) * self.mutation_rate
                    param.add_(noise)

    def evaluate_and_select(self, candidates: List[nn.Module], task_function) -> Tuple[nn.Module, float]:
        scores = []
        for i, candidate in enumerate(candidates):
            try:
                score = task_function(candidate)
            except Exception:
                score = 0.0
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_brain = candidates[best_idx]
        
        prev_score = self.improvement_history[-1] if self.improvement_history else 0.0
        improvement = best_score - prev_score
            
        self.improvement_history.append(best_score)
        self.generation += 1
        
        if improvement > 0.001:
            self.current_best_brain = best_brain
            
        return self.current_best_brain, best_score

    def get_status(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "current_score": self.improvement_history[-1] if self.improvement_history else 0.0
        }