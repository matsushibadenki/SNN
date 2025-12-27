# ファイルパス: snn_research/cognitive_architecture/meta_cognitive_snn.py
# 日本語タイトル: メタ認知SNNモジュール
# 役割: 自身の推論結果をモニタリングし、System 2（世界モデル）の起動要否を判断する。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class MetaCognitiveSNN(nn.Module):
    """
    自己監視モジュール。
    - Uncertainty Monitoring: エントロピーに基づく不確実性の検知
    - Surprise Detection: 予測と現実の乖離（驚き）の検知
    """
    def __init__(self, d_model: int, uncertainty_threshold: float = 1.5, surprise_threshold: float = 0.5):
        super().__init__()
        self.d_model = d_model
        # [Fix] Renamed/Aliased to match usage
        self.entropy_threshold = uncertainty_threshold
        # [Fix] Added missing attribute
        self.surprise_threshold = surprise_threshold
        
        # 簡易的な「確信度評価ネットワーク」
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def monitor_system1_output(self, logits: torch.Tensor) -> Dict[str, Any]:
        """
        System 1 (直感) の出力を監視する。
        """
        # 1. エントロピー計算 (不確実性の尺度)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # 2. 閾値判定
        is_uncertain = entropy.item() > self.entropy_threshold
        
        return {
            "entropy": entropy.item(),
            "trigger_system2": is_uncertain,
            "max_prob": probs.max().item()
        }

    def evaluate_surprise(self, predicted_state: torch.Tensor, actual_state: torch.Tensor) -> float:
        """
        予測された未来と、実際の観測との「驚き（Surprise）」を計算する。
        """
        # MSE (Mean Squared Error) を基本とするが、
        # 実際にはKLダイバージェンスや自由エネルギー(Free Energy)を用いる
        mse = F.mse_loss(predicted_state, actual_state)
        return mse.item()