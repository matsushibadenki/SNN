# ファイルパス: snn_research/adaptive/intrinsic_motivator.py
# 日本語タイトル: Intrinsic Motivator (Type Safe)

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class IntrinsicMotivator(nn.Module):
    running_error_mean: torch.Tensor

    def __init__(self, config: Dict[str, Any] = {}):
        super().__init__()
        self.curiosity_weight = config.get("curiosity_weight", 1.0)
        self.empowerment_weight = config.get("empowerment_weight", 0.5)
        self.decay_rate = config.get("decay_rate", 0.99)

        self.register_buffer("running_error_mean", torch.tensor(0.1))

        logger.info("🧠 Intrinsic Motivator initialized.")

    def compute_reward(
        self,
        predicted_state: torch.Tensor,
        actual_state: torch.Tensor,
        action_impact: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            prediction_error = F.mse_loss(
                predicted_state, actual_state, reduction='none').mean(dim=-1)
            batch_error = prediction_error.mean()

            self.running_error_mean = self.running_error_mean * \
                self.decay_rate + batch_error * (1 - self.decay_rate)

            curiosity = torch.relu(
                batch_error - self.running_error_mean) * 10.0

        empowerment = 0.0
        if action_impact is not None:
            empowerment = action_impact.norm(p=2, dim=-1).mean()

        total_reward = (self.curiosity_weight * curiosity) + \
            (self.empowerment_weight * empowerment)

        return total_reward

    def get_stats(self) -> Dict[str, float]:
        return {
            "baseline_error": self.running_error_mean.item()
        }
