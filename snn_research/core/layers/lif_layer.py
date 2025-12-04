# ファイルパス: snn_research/core/layers/lif_layer.py
# タイトル: Leaky Integrate-and-Fire (LIF) SNNレイヤー
# 機能説明: 
#   Project SNN4のロードマップ (Phase 4, P4-4) に基づく、
#   AbstractSNNLayer (P4-1) を継承する具象LIFレイヤー。
#
#   (ダミー実装の解消):
#   - P1-1 との連携を強化。
#   - build() メソッドで学習規則 (PredictiveCodingRule) を
#     インスタンス化する際に、自身の名前 (self.name) を
#     'layer_name' として渡します。

import logging
from typing import Dict, Any, Optional, Tuple, cast
import torch
import torch.nn as nn
from torch import Tensor 
from snn_research.core.synapse_dynamics import apply_probabilistic_transmission # 追加

# P4-1 (抽象SNNレイヤー) と P1 のモジュールをインポート
try:
    from .abstract_snn_layer import AbstractSNNLayer
    from ...layers.abstract_layer import LayerOutput
    from ...config.learning_config import BaseLearningConfig
    from ..learning_rule import Parameters
    from ..learning_rules.predictive_coding_rule import PredictiveCodingRule
except ImportError:
    # (mypy フォールバック)
    LayerOutput = Dict[str, Tensor] # type: ignore[misc]
    BaseLearningConfig = Any # type: ignore[misc, assignment]
    Parameters = Any # type: ignore[misc, assignment]
    from abc import ABC, abstractmethod
    class AbstractLayer(nn.Module, ABC): # type: ignore[no-redef, misc]
         def __init__(self, i: Any, o: Any, l: Any, n: str) -> None: super().__init__()
    class AbstractSNNLayer(AbstractLayer): # type: ignore[no-redef, misc]
        def __init__(self, i: Any, o: Any, l: Any, n: str) -> None: 
            super().__init__(i, o, l, n)
        def build(self) -> None: raise NotImplementedError
        def forward(self, i: Tensor, s: Dict[str, Tensor]) -> LayerOutput:
            raise NotImplementedError
        def reset_state(self) -> None: pass
        params: Parameters = []
        learning_config: Optional[BaseLearningConfig] = None
        learning_rule: Any = None
    PredictiveCodingRule = Any # type: ignore[misc, assignment]

# ロガーの設定
logger: logging.Logger = logging.getLogger(__name__)

@torch.jit.script
def lif_update(
    inputs: Tensor, 
    V: Tensor, 
    W: Tensor, 
    b: Tensor, 
    decay: float, 
    threshold: float
) -> Tuple[Tensor, Tensor]:
    """ P4-4: 単一ステップのLIFダイナミクス (PyTorch 実装) """
    I_t: Tensor = nn.functional.linear(inputs, W, b)
    V_leaked: Tensor = V * decay
    V_new: Tensor = V_leaked + I_t
    spikes: Tensor = (V_new > threshold).float()
    V_reset: Tensor = V_new - (spikes * threshold)
    return V_reset, spikes


class LIFLayer(AbstractSNNLayer):
    """
    P4-4: Leaky Integrate-and-Fire (LIF) レイヤー (PyTorch実装)。
    """

    def __init__(
        self, 
        input_features: int, 
        neurons: int,
        learning_config: Optional[BaseLearningConfig] = None,
        name: str = "LIFLayer",
        decay: float = 0.95, 
        threshold: float = 1.0,
        # --- ▼ 追加 ▼ ---
        synaptic_reliability: float = 1.0, # デフォルトは1.0 (ノイズなし)
        # --- ▲ 追加 ▲ ---
    ) -> None:
        
        dummy_shape: Tuple[int, ...] = (0,)
        # (name は AbstractLayer の __init__ で設定される)
        super().__init__(dummy_shape, dummy_shape, learning_config, name)
        
        self.decay: float = decay
        self.threshold: float = threshold
        
        self.synaptic_reliability = synaptic_reliability
        
        self._input_features: int = input_features
        self._neurons: int = neurons
        
        self.W: nn.Parameter = nn.Parameter(
            torch.empty(self._neurons, self._input_features), 
            requires_grad=False
        )
        self.b: nn.Parameter = nn.Parameter(
            torch.empty(self._neurons), 
            requires_grad=False
        )
        
        self.membrane_potential: Optional[Tensor] = None


    def build(self) -> None:
        """
        (P2-1) パラメータを初期化し、(P1-4) 学習規則をセットアップします。
        """
        if logger:
            logger.debug(f"Building layer: {self.name}")
            
        nn.init.kaiming_uniform_(self.W, a=0.01)
        nn.init.zeros_(self.b)
        
        # P1-4: 学習可能なパラメータとして登録
        self.params = [self.W, self.b]
        
        # P1-4: 学習規則のインスタンス化
        if self.learning_config:
            # (P1-1 の PC ルールをデフォルトで使用)
            rule_cls: Any = PredictiveCodingRule
            
            # --- ダミー実装の解消 (P1-1 / P4-4 連携) ---
            # P1-3 の設定を取得
            rule_kwargs: Dict[str, Any] = self.learning_config.to_dict()
            # P1-4 (AbstractLearningRule) のため、レイヤー名を渡す
            rule_kwargs['layer_name'] = self.name
            
            self.learning_rule = rule_cls(
                self.params, 
                **rule_kwargs
            )
            
        self.built = True

    def _init_state(self, batch_size: int, device: torch.device) -> None:
        if logger:
            logger.debug(f"Initializing state for {self.name} with batch size {batch_size}")
        
        self.membrane_potential = torch.zeros(
            (batch_size, self._neurons), device=device
        )

    def forward(
        self, 
        inputs: Tensor, 
        model_state: Dict[str, Tensor]
    ) -> LayerOutput:
        if not self.built:
            raise RuntimeError(f"Layer {self.name} has not been built.")
        
        batch_size: int = inputs.shape[0]

        if self.membrane_potential is None:
            self._init_state(batch_size, inputs.device)
        
        V_t_minus_1: Tensor = cast(Tensor, self.membrane_potential)

        # --- ▼ 修正: シナプスゆらぎの適用 ▼ ---
        # 毎ステップ異なるゆらぎを与えるため、forward内でマスクを生成
        effective_W = apply_probabilistic_transmission(
            self.W, 
            reliability=self.synaptic_reliability,
            training=self.training # 学習時のみ適用するか、常時適用するか
        )
        # --- ▲ 修正 ▲ ---

        V_t: Tensor
        spikes_t: Tensor
        # lif_update には effective_W を渡す
        V_t, spikes_t = lif_update(
            inputs, V_t_minus_1, effective_W, self.b, self.decay, self.threshold
        )
        
        self.membrane_potential = V_t
        
        return {
            'activity': spikes_t, 
            'membrane_potential': V_t
        }

    def reset_state(self) -> None:
        if logger:
            logger.debug(f"Resetting state for {self.name}")
        self.membrane_potential = None
