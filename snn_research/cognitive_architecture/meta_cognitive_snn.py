# ファイルパス: snn_research/cognitive_architecture/meta_cognitive_snn.py
# 日本語タイトル: Meta-Cognitive Monitor v1.4 (Methods Added)
# 修正内容: デモ動作に必要な monitor_system1_output と evaluate_surprise を追加。

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MetaCognitiveSNN(nn.Module):
    """
    自己の状態を監視し、制御するメタ認知モジュール。
    System 1の出力分布からエントロピー（不確実性）を計算し、System 2への切り替えを判断する。
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

        # 設定の読み込み (デフォルト値を設定)
        self.patience = config.get("patience", 10)
        self.sensitivity = config.get("sensitivity", 0.1)
        self.breakthrough_threshold = config.get("breakthrough_threshold", 0.9)
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.6) # エントロピー閾値

        # d_modelは将来的な拡張のために保持（現在は未使用）
        self.d_model = d_model

        self.error_history: List[float] = []

        logger.info("🧠 Meta-Cognitive System v1.4 initialized.")

    def monitor(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """学習時のパフォーマンス監視（既存メソッド）"""
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

    def monitor_system1_output(self, logits: torch.Tensor) -> Dict[str, Any]:
        """
        [New] System 1 (直感) の出力を監視し、不確実性(エントロピー)を計算する。
        エントロピーが高い場合、System 2 (熟考) のトリガーを発行する。
        
        Args:
            logits: 出力ロジット (Batch, NumClasses)
        """
        # 確率分布へ変換
        probs = F.softmax(logits, dim=-1)
        
        # エントロピー計算: -sum(p * log(p))
        # log(0)回避のために微小値を加算
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
        
        # 不確実性の判定
        trigger = entropy > self.uncertainty_threshold
        
        # 内部状態の更新 (簡易版)
        if trigger:
            self.frustration = torch.clamp(self.frustration + 0.05, 0.0, 1.0)
            self.confidence = torch.clamp(self.confidence - 0.05, 0.0, 1.0)
        else:
            self.confidence = torch.clamp(self.confidence + 0.02, 0.0, 1.0)
            
        return {
            "entropy": entropy,
            "trigger_system2": trigger,
            "confidence": self.confidence.item()
        }

    def evaluate_surprise(self, predicted: torch.Tensor, actual: torch.Tensor) -> float:
        """
        [New] 予測と現実の乖離 (Surprise / Prediction Error) を評価する。
        """
        with torch.no_grad():
            # 平均二乗誤差 (MSE) をSurpriseとする
            mse = F.mse_loss(predicted, actual).item()
        
        # Surpriseが大きい場合、学習率や注意力を高めるシグナルとして機能する
        if mse > 1.0:
            self.focus_level = torch.clamp(self.focus_level + 1.0, 1.0, 10.0)
            
        return mse

    def should_trigger_intervention(self) -> bool:
        return bool((self.frustration > 0.8).item())

    def reset_state(self):
        self.frustration.zero_()
        self.focus_level.fill_(1.0)
        self.error_history.clear()