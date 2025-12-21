# ファイルパス: snn_research/core/network.py
# 日本語タイトル: 抽象ネットワークモデル (リファクタ版)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import torch
import torch.nn as nn
from torch import Tensor
from snn_research.layers.abstract_layer import AbstractLayer

logger = logging.getLogger(__name__)

class AbstractNetwork(nn.Module, ABC):
    def __init__(self, layers: Optional[List[AbstractLayer]] = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers or [])
        self.built = False
        self._sync_layer_map()

    def _sync_layer_map(self) -> None:
        self.layer_map = {layer.name: layer for layer in self.layers if hasattr(layer, 'name')}

    def add_layer(self, layer: AbstractLayer) -> None:
        if layer.name in self.layer_map:
            raise ValueError(f"Layer {layer.name} exists.")
        self.layers.append(layer)
        self._sync_layer_map()

    def build_model(self) -> None:
        for layer in self.layers:
            if hasattr(layer, 'build'): layer.build()
        self.built = True

    @abstractmethod
    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Dict[str, Tensor]:
        pass

    def update_model(self, inputs: Tensor, targets: Optional[Tensor], model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """順伝播後の局所学習則を適用。"""
        if not self.built: raise RuntimeError("Call build_model() first.")
        
        all_metrics = {}
        current_input = inputs
        
        for layer in self.layers:
            metrics = layer.update_local(current_input, targets, model_state)
            all_metrics.update({f"{layer.name}_{k}": v for k, v in metrics.items()})
            
            # 次の層への入力を model_state から取得
            current_input = model_state.get(f'{layer.name}_output', current_input)
            
        return all_metrics
