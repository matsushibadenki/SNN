# ファイルパス: snn_research/cognitive_architecture/meta_cognitive_snn.py
# 日本語タイトル: Meta-Cognitive Monitor v1.3 (DI Fix)
# 修正内容: DIコンテナからの d_model 引数を受け入れるように修正。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MetaCognitiveSNN(nn.Module):
    """
    自己の状態を監視し、制御するメタ認知モジュール。
    """
    confidence: torch.Tensor
    frustration: torch.Tensor
    focus_level: torch.Tensor

    def __init__(
        self,
        config: Dict[str, Any] = {},
        d_model: Optional[int] = None  # Added for DI compatibility
    ):
        super().__init__()

        self.register_buffer("confidence", torch.tensor(0.5))
        self.register_buffer("frustration", torch.tensor(0.0))
        self.register_buffer("focus_level", torch.tensor(1.0))

        self.patience = config.get("patience", 10)
        self.sensitivity = config.get("sensitivity", 0.1)
        self.breakthrough_threshold = config.get("breakthrough_threshold", 0.9)

        # d_modelは将来的な拡張のために保持（現在は未使用）
        self.d_model = d_model

        self.error_history: List[float] = []

        logger.info("🧠 Meta-Cognitive System v1.3 initialized.")

    def monitor(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        error = performance_metrics.get("error", 0.0)
        reward = performance_metrics.get("reward", 0.0)

        self.error_history.append(error)
        if len(self.error_history) > self.patience:
            self.error_history.pop(0)

        recent_avg_error = sum(self.error_history) / \
            len(self.error_history) if self.error_history else 0.0

        if recent_avg_error > 0.3:
            self.frustration = torch.clamp(
                self.frustration + self.sensitivity, 0.0, 1.0)
        else:
            self.frustration = torch.clamp(
                self.frustration - self.sensitivity, 0.0, 1.0)

        if reward > 0:
            self.confidence = torch.clamp(self.confidence + 0.05, 0.0, 1.0)
        elif error > 0.5:
            self.confidence = torch.clamp(self.confidence - 0.05, 0.0, 1.0)

        if self.frustration > self.breakthrough_threshold:
            self.focus_level = torch.tensor(100.0)
        else:
            self.focus_level = 1.0 + (self.frustration * 5.0)

        return {
            "confidence": self.confidence.item(),
            "frustration": self.frustration.item(),
            "focus_level": self.focus_level.item()
        }

    def should_trigger_intervention(self) -> bool:
        return bool((self.frustration > 0.8).item())

    def reset_state(self):
        self.frustration.zero_()
        self.focus_level.fill_(1.0)
        self.error_history.clear()
