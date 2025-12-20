# ファイルパス: snn_research/core/network.py
# 日本語タイトル: 抽象ネットワークモデル (修正版)
# 機能説明: 
#   AbstractLayer を組み合わせて構成されるネットワークの基底クラス。
#   各層の構築(build)や更新(update_model)を一括管理する。
#   
#   修正点:
#   - nn.Module を継承し、PyTorchパラメータ管理に対応。
#   - update_model 内で current_input を更新し、層間のデータフローを正しく実装。

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import torch
import torch.nn as nn
from torch import Tensor

# 絶対インポート
from snn_research.layers.abstract_layer import AbstractLayer

logger = logging.getLogger(__name__)

class AbstractNetwork(nn.Module, ABC):
    """
    局所学習則を持つレイヤー群を管理する抽象ネットワーククラス。
    nn.Moduleを継承しているため、OptimizerやState Dictと連携可能。
    """

    def __init__(self, layers: Optional[List[AbstractLayer]] = None) -> None:
        super().__init__()
        # nn.ModuleListを使用することで、PyTorchがサブモジュールとして認識・管理できる
        self.layers: nn.ModuleList = nn.ModuleList(layers if layers is not None else [])
        self.layer_map: Dict[str, AbstractLayer] = {}
        self.built: bool = False
        
        if layers:
            self._build_layer_map()

    def add_layer(self, layer: AbstractLayer) -> None:
        if self.built:
            logger.warning(f"Adding layer {layer.name} after model was built.")
        
        if layer.name in self.layer_map:
            raise ValueError(f"Duplicate layer name found: {layer.name}")
            
        self.layers.append(layer)
        self.layer_map[layer.name] = layer

    def _build_layer_map(self) -> None:
        self.layer_map.clear()
        for layer in self.layers:
            if isinstance(layer, AbstractLayer):
                if layer.name in self.layer_map:
                    raise ValueError(f"Duplicate layer name found: {layer.name}")
                self.layer_map[layer.name] = layer

    def build_model(self) -> None:
        """全てのレイヤーをビルドする"""
        logger.info("Building network model...")
        for layer in self.layers:
            if isinstance(layer, AbstractLayer):
                layer.build()
        self.built = True
        logger.info("Network build complete.")

    @abstractmethod
    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        順伝播を実行する。
        戻り値はモデルの状態や出力を含む辞書。
        """
        raise NotImplementedError

    def update_model(self, inputs: Tensor, targets: Optional[Tensor], model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        各レイヤーの update_local を呼び出し、ネットワーク全体を更新する。
        forward() 実行後に呼び出されることを想定しており、model_state に各層の出力が格納されている必要がある。
        """
        if not self.built:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        all_metrics: Dict[str, Tensor] = {}
        
        # 最初の層への入力
        current_input: Tensor = inputs
        
        for layer in self.layers:
            if isinstance(layer, AbstractLayer):
                # 各レイヤーで局所的な更新を実行
                # current_input はこの層への入力 (前段の出力)
                layer_metrics = layer.update_local(current_input, targets, model_state)
                
                for metric_name, metric_value in layer_metrics.items():
                    all_metrics[f"{layer.name}_{metric_name}"] = metric_value
                
                # 次の層への入力を準備 (model_state からこの層の出力を取得)
                output_key = f'{layer.name}_output'
                if output_key in model_state:
                    current_input = model_state[output_key]
                    # 注意: model_stateに記録されていない場合、current_inputは更新されない
                    # (ResNetのような分岐がある場合はより複雑なロジックが必要だが、ここでは直列を想定)
            
        return all_metrics

    def get_parameters(self) -> List[nn.Parameter]:
        """全レイヤーの学習可能パラメータを取得"""
        # nn.Module を継承しているため parameters() が使えるが、
        # AbstractLayer.params に明示的に登録されたものだけを返す場合の実装
        all_params: List[nn.Parameter] = []
        for layer in self.layers:
            if isinstance(layer, AbstractLayer):
                all_params.extend(layer.params)
        return all_params