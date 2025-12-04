# ファイルパス: snn_research/core/layers/abstract_snn_layer.py
# タイトル: 抽象SNN（スパイキングニューラルネットワーク）レイヤー (PyTorch準拠)
# 機能説明: 
#   Project SNN4のロードマップ (P4-1) に基づく、
#   SNNに特有の機能（状態リセット）を持つ抽象基底クラス。
#
#   (ダミー実装の解消):
#   - Generic[Tensor] および TypeVar('Tensor') を削除。
#   - P2-1 の 'AbstractLayer' (nn.Module 準拠) を継承します。
#   - 'torch.Tensor' を直接使用します。

from abc import abstractmethod, ABC
from typing import Dict, Any, Optional

# --- ダミー実装の解消 (PyTorch のインポート) ---
import torch.nn as nn
from torch import Tensor

# P2-1 (抽象レイヤー) をインポート
try:
    from ...layers.abstract_layer import AbstractLayer, LayerOutput
    from ...config.learning_config import BaseLearningConfig
except ImportError:
    # (mypy フォールバック)
    LayerOutput = Dict[str, Tensor] # type: ignore[misc]
    # 修正 (エラー 13): mypy [assignment] 競合を無視
    BaseLearningConfig = Any # type: ignore[misc, assignment]
    
    class AbstractLayer(nn.Module, ABC): # type: ignore[no-redef, misc]
        def __init__(
            self, i: Any, o: Any, l: Any, n: str
        ) -> None: super().__init__()
        
        # 修正 (エラー 14): タイプミスを修正
        @abstractmethod
        def build(self) -> None: raise NotImplementedError
        
        @abstractmethod
        def forward(
            self, i: Tensor, s: Dict[str, Tensor]
        ) -> LayerOutput:
            raise NotImplementedError


# 修正: Generic[Tensor] を削除し、AbstractLayer を継承
class AbstractSNNLayer(AbstractLayer):
    """
    P4-1: SNNレイヤーのための抽象基底クラス。
    """

    def __init__(
        self, 
        input_shape: Any,
        output_shape: Any,
        learning_config: Optional[BaseLearningConfig] = None,
        name: str = "AbstractSNNLayer"
    ) -> None:
        """
        抽象SNNレイヤーを初期化します。
        """
        super().__init__(input_shape, output_shape, learning_config, name)
        
        # P4-1: SNNの内部状態 (例: 膜電位)
        self.membrane_potential: Optional[Tensor] = None

    @abstractmethod
    def build(self) -> None:
        """
        (AbstractLayerから継承)
        レイヤーのパラメータ（重み）と内部状態（膜電位）を初期化します。
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, 
        inputs: Tensor, 
        model_state: Dict[str, Tensor]
    ) -> LayerOutput:
        """
        (AbstractLayerから継承)
        単一の時間ステップ (t) における順伝播を実行します。
        """
        raise NotImplementedError

    @abstractmethod
    def reset_state(self) -> None:
        """
        P4-1: レイヤーの内部状態 (膜電位など) をリセットします。
        """
        raise NotImplementedError
