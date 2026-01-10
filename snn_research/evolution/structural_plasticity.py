# ファイルパス: snn_research/evolution/structural_plasticity.py
# 日本語タイトル: Structural Plasticity Engine (Type Safe)

import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class StructuralPlasticity(nn.Module):
    def __init__(self, model: nn.Module, config: Dict[str, Any] = {}):
        super().__init__()
        self.model = model
        self.pruning_rate = config.get("pruning_rate", 0.1)
        self.growth_rate = config.get("growth_rate", 0.1)
        self.noise_std = config.get("noise_std", 0.01)

        logger.info("🧬 Structural Plasticity Engine initialized.")

    def evolve_structure(self) -> Dict[str, int]:
        total_pruned = 0
        total_grown = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    weights = module.weight.data
                    importance = weights.abs()
                    threshold = torch.quantile(importance, self.pruning_rate)

                    mask = importance > threshold
                    pruned_weights = weights * mask

                    # Explicit int cast for mypy
                    num_pruned = int((weights.numel() - mask.sum()).item())

                    dead_mask = ~mask
                    new_connections = torch.randn_like(
                        weights) * self.noise_std
                    final_weights = pruned_weights + \
                        (new_connections * dead_mask.float())

                    module.weight.data = final_weights

                    total_pruned += num_pruned
                    total_grown += num_pruned

        return {
            "pruned": total_pruned,
            "grown": total_grown
        }

    def inject_noise(self, intensity: float = 0.01):
        for param in self.model.parameters():
            if param.requires_grad:
                with torch.no_grad():
                    noise = torch.randn_like(param) * intensity
                    param.add_(noise)
