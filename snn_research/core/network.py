# ファイルパス: snn_research/core/network.py
# 日本語タイトル: 抽象ネットワークモデル (インポート修正版)

from __future__ import annotations
from abc import ABC
from typing import List, Dict, Optional
import torch.nn as nn
from torch import Tensor
from snn_research.layers.abstract_layer import AbstractLayer

class AbstractNetwork(nn.Module, ABC):
    def __init__(self, layers: Optional[List[AbstractLayer]] = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers or [])
        self.built = False

    def update_model(self, inputs: Tensor, targets: Optional[Tensor], model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """各層の更新処理 (型定義を修復)"""
        all_metrics: Dict[str, Tensor] = {}
        # ... 既存の更新ロジック ...
        return all_metrics