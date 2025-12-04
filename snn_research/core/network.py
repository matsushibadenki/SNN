# ファイルパス: snn_research/core/network.py
# タイトル: 抽象ネットワークモデルインターフェース (PyTorch準拠)
# 機能説明: 
#   Project SNN4のロードマップ (P2-2) に基づく、
#   ネットワーク全体のアーキテクチャを抽象化するインターフェース。
#
#   (ダミー実装の解消):
#   - Generic[Tensor] および TypeVar('Tensor') を削除。
#   - PyTorch の 'torch.Tensor' と 'torch.nn.Parameter' を使用します。
#   - (AbstractLayer が nn.Module を継承するため、このクラスは
#      nn.Module を継承する必要はないが、管理は行う)

from abc import ABC, abstractmethod
from typing import (
    List, Dict, Any, Optional, Iterable
)
import logging

# --- ダミー実装の解消 (PyTorch のインポート) ---
import torch.nn as nn
from torch import Tensor

# P2-1 (抽象レイヤー) と P1-4 (学習規則) のモジュールをインポート
try:
    from snn_research.core.layers.abstract_layer import (
        AbstractLayer, LayerOutput, UpdateMetrics
    )
    from .learning_rule import Parameters
except ImportError:
    # (mypy フォールバック)
    # 修正 (エラー 7): Generic を削除
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

# ロガーの設定
logger: logging.Logger = logging.getLogger(__name__)

# 修正 (エラー 8): Generic[Tensor] を削除
class AbstractNetwork(ABC):
    """
    P2-2: BPフリー学習モデルのための抽象ネットワーク。
    """

    # 修正 (エラー 9): [Tensor] を削除
    def __init__(self, layers: Optional[List[AbstractLayer]] = None) -> None:
        """
        ネットワークを初期化します。
        """
        # 修正: [Tensor] を削除
        self.layers: List[AbstractLayer] = layers if layers is not None else []
        
        # 修正: [Tensor] を削除
        self.layer_map: Dict[str, AbstractLayer] = {}
        
        self.built: bool = False
        
        if layers:
            self._build_layer_map()

    # 修正 (エラー 10): [Tensor] を削除
    def add_layer(self, layer: AbstractLayer) -> None:
        """
        ネットワークにレイヤーを追加します。
        """
        if self.built:
            if logger:
                logger.warning(f"Adding layer {layer.name} after model was built.")
        
        if layer.name in self.layer_map:
            raise ValueError(f"Duplicate layer name found: {layer.name}")
            
        self.layers.append(layer)
        self.layer_map[layer.name] = layer

    def _build_layer_map(self) -> None:
        """内部のレイヤーマップを構築します。"""
        self.layer_map.clear()
        for layer in self.layers:
            if layer.name in self.layer_map:
                raise ValueError(f"Duplicate layer name found: {layer.name}")
            self.layer_map[layer.name] = layer

    def build_model(self) -> None:
        """
        ネットワーク内のすべてのレイヤーをビルドします。
        """
        if logger:
            logger.info("Building network model...")
        for layer in self.layers:
            layer.build()
        self.built = True
        if logger:
            logger.info("Network build complete.")

    @abstractmethod
    def forward(
        self, 
        inputs: Tensor, 
        targets: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        ネットワーク全体の順伝播を実行します。
        """
        raise NotImplementedError

    def update_model(
        self, 
        inputs: Tensor, 
        targets: Optional[Tensor],
        model_state: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        P1-2 (ローカル学習) に基づき、すべての学習可能なレイヤーを更新します。
        """
        if not self.built:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        all_metrics: Dict[str, Tensor] = {}
        
        current_input: Tensor = inputs
        
        for layer in self.layers:
            layer_metrics = layer.update_local(
                current_input, 
                targets, 
                model_state
            )
            
            for metric_name, metric_value in layer_metrics.items():
                all_metrics[f"{layer.name}_{metric_name}"] = metric_value
            
            output_key: str = f'{layer.name}_output'
            if output_key in model_state and isinstance(model_state[output_key], dict):
                # (mypy) model_state は Dict[str, Tensor] だが、
                # 実際の値は LayerOutput (Dict[str, Tensor])
                layer_output: Dict[str, Tensor] = model_state[output_key] # type: ignore[assignment]
                if 'activity' in layer_output:
                     current_input = layer_output['activity']
            
        return all_metrics

    def get_parameters(self) -> Iterable[Parameters]:
        """ネットワーク全体の学習可能パラメータを取得します。"""
        all_params: List[Parameters] = []
        for layer in self.layers:
            all_params.append(layer.params)
        return all_params
