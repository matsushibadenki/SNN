# ファイルパス: snn_research/core/layers/lif_layer.py
# タイトル: LIF (Leaky Integrate-and-Fire) SNNレイヤー (高信頼性実装)
# 機能説明: ロードマップ Phase 4 に準拠。シナプス信頼性のシミュレーションと動的な学習規則のセットアップ。

import logging
from typing import Dict, Any, Optional, Tuple, cast, List, TYPE_CHECKING
import torch
import torch.nn as nn
from torch import Tensor 

from snn_research.core.layers.abstract_snn_layer import AbstractSNNLayer
from snn_research.layers.abstract_layer import LayerOutput
from snn_research.config.learning_config import BaseLearningConfig
from snn_research.core.learning_rule import Parameters
from snn_research.core.synapse_dynamics import apply_probabilistic_transmission

if TYPE_CHECKING:
    from snn_research.core.learning_rules.predictive_coding_rule import PredictiveCodingRule

logger = logging.getLogger(__name__)

@torch.jit.script
def lif_update_optimized(
    inputs: Tensor, 
    V: Tensor, 
    W: Tensor, 
    b: Tensor, 
    decay: float, 
    threshold: float
) -> Tuple[Tensor, Tensor]:
    """JITコンパイルされた高速なLIFダイナミクス。"""
    # 入力電流 (I = Wx + b)
    I_t = nn.functional.linear(inputs, W, b)
    
    # 電位更新 (V = V*decay + I)
    V_new = (V * decay) + I_t
    
    # 発火判定 (ハード閾値)
    spikes = (V_new >= threshold).float()
    
    # 膜電位リセット (Soft Reset: 閾値を引くことで残余電位を維持)
    V_reset = V_new - (spikes * threshold)
    
    return V_reset, spikes

class LIFLayer(AbstractSNNLayer):
    """具象LIFレイヤー実装。"""
    def __init__(
        self, 
        input_features: int, 
        neurons: int,
        learning_config: Optional[BaseLearningConfig] = None,
        name: str = "LIFLayer",
        decay: float = 0.95, 
        threshold: float = 1.0,
        synaptic_reliability: float = 1.0,
    ) -> None:
        super().__init__((input_features,), (neurons,), learning_config, name)
        
        self.decay = decay
        self.threshold = threshold
        self.synaptic_reliability = synaptic_reliability
        self._neurons = neurons
        
        # 学習則での手動更新を前提に requires_grad=False
        self.W = nn.Parameter(torch.empty(neurons, input_features), requires_grad=False)
        self.b = nn.Parameter(torch.empty(neurons), requires_grad=False)
        
        self.membrane_potential: Optional[Tensor] = None
        self.total_spikes = nn.Buffer(torch.tensor(0.0)) # BaseModelからの集計用

    def build(self) -> None:
        """初期化と学習規則の動的バインド。"""
        nn.init.kaiming_uniform_(self.W, a=0.01)
        nn.init.zeros_(self.b)
        self.params = [self.W, self.b]
        
        if self.learning_config:
            try:
                # 循環参照を回避しつつ、PredictiveCodingRuleをセットアップ
                from snn_research.core.learning_rules.predictive_coding_rule import PredictiveCodingRule
                rule_kwargs = self.learning_config.to_dict()
                rule_kwargs['layer_name'] = self.name
                self.learning_rule = PredictiveCodingRule(self.params, **rule_kwargs)
                logger.info(f"✅ {self.name}: Learning rule 'PredictiveCodingRule' assigned.")
            except ImportError:
                logger.warning(f"⚠️ {self.name}: PredictiveCodingRule not found. Running without local rule.")

        self.built = True

    def forward(self, inputs: Tensor, model_state: Dict[str, Tensor]) -> LayerOutput:
        if not self.built:
            self.build() # 未ビルドなら実行時にビルド

        if self.membrane_potential is None or self.membrane_potential.shape[0] != inputs.shape[0]:
            self.membrane_potential = torch.zeros(inputs.shape[0], self._neurons, device=inputs.device)

        # シナプスゆらぎの適用
        effective_W = apply_probabilistic_transmission(
            self.W, reliability=self.synaptic_reliability, training=self.training 
        )

        # LIF更新
        V_next, spikes = lif_update_optimized(
            inputs, self.membrane_potential, effective_W, self.b, self.decay, self.threshold
        )
        
        self.membrane_potential = V_next
        self.total_spikes += spikes.sum().detach()
        
        return {'activity': spikes, 'membrane_potential': V_next}

    def reset_state(self) -> None:
        self.membrane_potential = None
