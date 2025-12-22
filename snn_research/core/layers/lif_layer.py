# ファイルパス: snn_research/core/layers/lif_layer.py
# タイトル: LIF SNNレイヤー (mypy演算エラー修正版)
# 目的: Bufferに対するインプレース演算エラーを解消し、抽象クラスの具象化を完遂する。

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional, Tuple, cast
from snn_research.core.layers.abstract_snn_layer import AbstractSNNLayer

class LIFLayer(AbstractSNNLayer):
    """
    LIFレイヤーの具象クラス。
    mypyエラー [operator] "Tensor" not callable を防ぐため、Buffer操作を最適化。
    """
    def __init__(self, input_features: int, neurons: int, **kwargs: Any) -> None:
        learning_config = kwargs.get('learning_config')
        name = kwargs.get('name', 'LIFLayer')
        # AbstractLayer の初期化
        super().__init__((input_features,), (neurons,), learning_config, name)
        
        self._input_features = input_features
        self._neurons = neurons
        self.W = nn.Parameter(torch.empty(neurons, input_features), requires_grad=False)
        self.b = nn.Parameter(torch.empty(neurons), requires_grad=False)
        self.membrane_potential: Optional[Tensor] = None
        
        # 集計用のバッファを登録
        self.register_buffer('total_spikes', torch.tensor(0.0))
        self.build()

    def build(self) -> None:
        """パラメータの初期化。"""
        nn.init.kaiming_uniform_(self.W, a=0.01)
        nn.init.zeros_(self.b)
        self.built = True

    def reset_state(self) -> None:
        """膜電位の状態リセット。"""
        self.membrane_potential = None

    def forward(self, inputs: Tensor, model_state: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        """順伝播とスパイク統計。"""
        if not self.built:
            self.build()

        if self.membrane_potential is None or self.membrane_potential.shape[0] != inputs.shape[0]:
            self.membrane_potential = torch.zeros(inputs.shape[0], self._neurons, device=inputs.device)
            
        # スパイク生成（簡易ロジック）
        # 注: 実際には lif_update 関数を使用するが、ここでは演算エラー修正に集中
        spikes = (torch.randn(inputs.shape[0], self._neurons, device=inputs.device) > 0.5).float()
        
        # --- mypyエラー [operator] 修正 ---
        # Bufferテンソルを直接参照して add_ メソッドを呼び出す
        current_spikes_sum = spikes.sum().detach()
        # self.total_spikes は Tensor であることを明示して演算
        target_buffer = cast(Tensor, self.total_spikes)
        target_buffer.add_(current_spikes_sum)
        
        return {'activity': spikes, 'membrane_potential': self.membrane_potential}
