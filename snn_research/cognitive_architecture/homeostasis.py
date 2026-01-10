# ファイルパス: snn_research/cognitive_architecture/homeostasis.py
# 日本語タイトル: Homeostasis System (Type Safe)

import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Homeostasis(nn.Module):
    # Registered buffers types
    energy: torch.Tensor
    fatigue: torch.Tensor
    cycle_count: torch.Tensor

    def __init__(self, config: Dict[str, Any] = {}):
        super().__init__()

        self.max_energy = config.get("max_energy", 100.0)
        self.fatigue_rate = config.get("fatigue_rate", 0.5)
        self.recovery_rate = config.get("recovery_rate", 5.0)
        self.sleep_threshold = config.get("sleep_threshold", 80.0)

        self.register_buffer("energy", torch.tensor(self.max_energy))
        self.register_buffer("fatigue", torch.tensor(0.0))
        self.register_buffer("cycle_count", torch.tensor(0))

        logger.info("💓 Homeostasis System initialized.")

    def update(self, action_intensity: float = 1.0) -> Dict[str, float]:
        fatigue_increase = self.fatigue_rate * action_intensity
        self.fatigue = torch.clamp(self.fatigue + fatigue_increase, 0, 100)
        self.energy = torch.clamp(
            self.energy - (fatigue_increase * 0.5), 0, self.max_energy)
        return self.get_status()

    def rest(self) -> Dict[str, float]:
        self.fatigue = torch.clamp(self.fatigue - self.recovery_rate, 0, 100)
        self.energy = torch.clamp(
            self.energy + (self.recovery_rate * 0.2), 0, self.max_energy)
        return self.get_status()

    def check_needs(self) -> str:
        if self.fatigue > self.sleep_threshold:
            return "sleep"
        elif self.energy < 20.0:
            return "recharge"
        else:
            return "explore"

    def new_day(self):
        self.cycle_count += 1
        logger.info(f"🌅 Day {self.cycle_count.item()} started. Fatigue reset.")

    def get_status(self) -> Dict[str, float]:
        return {
            "energy": self.energy.item(),
            "fatigue": self.fatigue.item(),
            "cycle": float(self.cycle_count.item())  # Explicit cast
        }
