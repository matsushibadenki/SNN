# ファイルパス: snn_research/core/network.py
# Title: 抽象ネットワークモデルインターフェース (PyTorch準拠・修正版)
# Description:
#   mypyエラー [import-not-found] を修正。

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterable
import logging
import torch.nn as nn
from torch import Tensor

# --- ▼ 修正: 正しいパスからのインポートに変更 ▼ ---
try:
    from snn_research.layers.abstract_layer import (
        AbstractLayer, LayerOutput, UpdateMetrics
    )
    from .learning_rule import Parameters
except ImportError:
    # (mypy フォールバック)
    Parameters = Iterable[nn.Parameter] # type: ignore[misc]
    LayerOutput = Dict[str, Tensor] # type: ignore[misc]
    UpdateMetrics = Dict[str, Tensor] # type: ignore[misc]
    
    class AbstractLayer(nn.Module): # type: ignore[no-redef, misc]
        name: str = "dummy"
        def build(self) -> None: pass
        def forward(
            self, i: Tensor, s: Dict[str, Tensor]
        ) -> LayerOutput: 
            return {}
        def update_local(
            self, i: Tensor, t: Optional[Tensor], s: Dict[str, Tensor]
        ) -> UpdateMetrics: 
            return {}
        params: Parameters = []
# --- ▲ 修正 ▲ ---

logger: logging.Logger = logging.getLogger(__name__)

class AbstractNetwork(ABC):
    """
    P2-2: BPフリー学習モデルのための抽象ネットワーク。
    """

    def __init__(self, layers: Optional[List[AbstractLayer]] = None) -> None:
        self.layers: List[AbstractLayer] = layers if layers is not None else []
        self.layer_map: Dict[str, AbstractLayer] = {}
        self.built: bool = False
        if layers:
            self._build_layer_map()

    def add_layer(self, layer: AbstractLayer) -> None:
        if self.built:
            if logger:
                logger.warning(f"Adding layer {layer.name} after model was built.")
        if layer.name in self.layer_map:
            raise ValueError(f"Duplicate layer name found: {layer.name}")
        self.layers.append(layer)
        self.layer_map[layer.name] = layer

    def _build_layer_map(self) -> None:
        self.layer_map.clear()
        for layer in self.layers:
            if layer.name in self.layer_map:
                raise ValueError(f"Duplicate layer name found: {layer.name}")
            self.layer_map[layer.name] = layer

    def build_model(self) -> None:
        if logger:
            logger.info("Building network model...")
        for layer in self.layers:
            layer.build()
        self.built = True
        if logger:
            logger.info("Network build complete.")

    @abstractmethod
    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Dict[str, Tensor]:
        raise NotImplementedError

    def update_model(self, inputs: Tensor, targets: Optional[Tensor], model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not self.built:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        all_metrics: Dict[str, Tensor] = {}
        current_input: Tensor = inputs
        
        for layer in self.layers:
            layer_metrics = layer.update_local(current_input, targets, model_state)
            
            for metric_name, metric_value in layer_metrics.items():
                all_metrics[f"{layer.name}_{metric_name}"] = metric_value
            
            output_key: str = f'{layer.name}_output'
            if output_key in model_state and isinstance(model_state[output_key], dict):
                layer_output: Dict[str, Tensor] = model_state[output_key] # type: ignore[assignment]
                if 'activity' in layer_output:
                     current_input = layer_output['activity']
            
        return all_metrics

    def get_parameters(self) -> Iterable[Parameters]:
        all_params: List[Parameters] = []
        for layer in self.layers:
            all_params.append(layer.params)
        return all_params
