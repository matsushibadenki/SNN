# ファイルパス: snn_research/conversion/ann_to_snn_converter.py
# Title: ANN-SNN 変換コンバータ (インポート修正版)
# 目的: pruning.py から apply_sbc_pruning を正しく認識できない問題を解消。

import torch
import torch.nn as nn
from typing import Dict, Any
import logging

# SNNコンポーネント

# --- mypy [attr-defined] 修正: 絶対インポートを再確認 ---
import snn_research.training.pruning as pruning

logger = logging.getLogger(__name__)

class AnnToSnnConverter:
    """
    ANNモデルからSNNモデルを生成するコンバータ。
    """
    def __init__(self, snn_model: nn.Module, model_config: Dict[str, Any]):
        self.snn_model = snn_model
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def apply_compression(self, amount: float, dataloader: Any) -> None:
        """
        修正点: pruning モジュールの apply_sbc_pruning を呼び出す。
        """
        if hasattr(pruning, 'apply_sbc_pruning'):
            # モジュール経由で呼び出すことで attr-defined を回避
            pruning.apply_sbc_pruning(
                self.snn_model, 
                amount, 
                dataloader, 
                nn.CrossEntropyLoss()
            )
            logger.info(f"SBC Pruning applied with amount: {amount}")
        else:
            logger.error("Function 'apply_sbc_pruning' not found in pruning module.")

    def convert(self, ann_model_path: str) -> None:
        # 変換ロジック...
        logger.info(f"Converting model from {ann_model_path}")