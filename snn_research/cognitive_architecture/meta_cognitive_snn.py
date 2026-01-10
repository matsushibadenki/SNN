# ファイルパス: snn_research/cognitive_architecture/meta_cognitive_snn.py
# 日本語タイトル: Meta-Cognitive Monitor v1.1 (Eureka Mode)
# 目的・内容:
#   [Update] フラストレーションが閾値を超えた際、学習率を劇的にブーストする
#   「ブレイクスルー（Eureka）機能」を追加し、停滞を打破できるように改良。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MetaCognitiveSNN(nn.Module):
    """
    自己の状態を監視し、制御するメタ認知モジュール。
    """

    def __init__(self, config: Dict[str, Any] = {}):
        super().__init__()

        # 状態変数
        self.register_buffer("confidence", torch.tensor(0.5))
        self.register_buffer("frustration", torch.tensor(0.0))
        self.register_buffer("focus_level", torch.tensor(1.0))

        # パラメータ
        self.patience = config.get("patience", 10)
        self.sensitivity = config.get("sensitivity", 0.1)
        self.breakthrough_threshold = config.get("breakthrough_threshold", 0.9)

        self.error_history = []

        logger.info(
            "🧠 Meta-Cognitive System v1.1 (Eureka Enabled) initialized.")

    def monitor(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        現在のパフォーマンスを評価し、内部状態を更新する。
        """
        error = performance_metrics.get("error", 0.0)
        reward = performance_metrics.get("reward", 0.0)

        # 1. 履歴更新
        self.error_history.append(error)
        if len(self.error_history) > self.patience:
            self.error_history.pop(0)

        # 2. フラストレーション計算
        recent_avg_error = sum(self.error_history) / \
            len(self.error_history) if self.error_history else 0.0

        if recent_avg_error > 0.3:
            self.frustration = torch.clamp(
                self.frustration + self.sensitivity, 0.0, 1.0)
        else:
            self.frustration = torch.clamp(
                self.frustration - self.sensitivity, 0.0, 1.0)

        # 3. 自信度更新
        if reward > 0:
            self.confidence = torch.clamp(self.confidence + 0.05, 0.0, 1.0)
        elif error > 0.5:
            self.confidence = torch.clamp(self.confidence - 0.05, 0.0, 1.0)

        # 4. 制御シグナルの生成 (Focus & Breakthrough)
        if self.frustration > self.breakthrough_threshold:
            # ✨ EUREKA MODE: Radical plasticity boost
            # 限界を超えた不満は、革命的な変化（学習率の爆発的増大）を引き起こす
            self.focus_level = torch.tensor(100.0)
        else:
            # Standard Focus: Linear increase
            self.focus_level = 1.0 + (self.frustration * 5.0)

        return {
            "confidence": self.confidence.item(),
            "frustration": self.frustration.item(),
            "focus_level": self.focus_level.item()
        }

    def should_trigger_intervention(self) -> bool:
        return self.frustration > 0.8

    def reset_state(self):
        self.frustration.zero_()
        self.focus_level.fill_(1.0)
        self.error_history.clear()
