# ファイルパス: snn_research/cognitive_architecture/meta_cognitive_snn.py
# Title: Meta-Cognitive SNN v2.2 - Correct Normalization
# Description: エントロピー計算の正規化ロジックを修正し、System 2トリガーを正常化。

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class MetaCognitiveSNN(nn.Module):
    """
    メタ認知モニター。
    - Uncertainty (Entropy): 何をすべきかわからない度合い。 -> System 2 Trigger
    """
    def __init__(
        self, 
        d_model: int = 128,
        uncertainty_threshold: float = 0.6,
        surprise_threshold: float = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.uncertainty_threshold = uncertainty_threshold
        self.surprise_threshold = surprise_threshold
        
        # 状態評価用の軽量ネットワーク
        self.value_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.uncertainty_history: List[float] = []
        self.surprise_history: List[float] = []
        
        logger.info("🧐 Meta-Cognitive SNN v2.2 initialized.")

    def monitor_system1_output(self, logits: torch.Tensor) -> Dict[str, Any]:
        """
        System 1 (SFormer) の出力分布(logits)を監視し、不確実性を評価する。
        """
        # logits: (Batch, ActionDim)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # エントロピー計算: H(p) = -sum(p * log p)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # 正規化: 0.0 - 1.0 の範囲にするため、最大エントロピー log(N) で割る
        num_classes = logits.size(-1)
        max_entropy = math.log(num_classes) if num_classes > 1 else 1.0
        normalized_entropy = entropy / max_entropy
        
        self.uncertainty_history.append(normalized_entropy.item())
        if len(self.uncertainty_history) > 100: self.uncertainty_history.pop(0)
        
        # 閾値を超えたらSystem 2 (熟慮) を要請
        trigger_system2 = normalized_entropy.item() > self.uncertainty_threshold
        
        return {
            "entropy": normalized_entropy.item(),
            "trigger_system2": trigger_system2,
            "confidence": 1.0 - normalized_entropy.item()
        }

    def evaluate_surprise(self, predicted_state: torch.Tensor, actual_state: torch.Tensor) -> float:
        """予測誤差(Surprise)の計算"""
        with torch.no_grad():
            mse = F.mse_loss(predicted_state, actual_state).item()
            is_surprised = mse > self.surprise_threshold
            
            self.surprise_history.append(mse)
            if len(self.surprise_history) > 100: self.surprise_history.pop(0)
            
            if is_surprised:
                logger.info(f"😲 Surprise detected! (Error: {mse:.4f})")
            return mse

    def update_metadata(self, loss: float, step_time: float, accuracy: float) -> None:
        self.surprise_history.append(loss)
        if len(self.surprise_history) > 100: self.surprise_history.pop(0)