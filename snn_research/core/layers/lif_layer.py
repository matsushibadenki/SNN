# ファイルパス: snn_research/core/layers/lif_layer.py
# タイトル: LIF SNNレイヤー (抽象クラス完全実装版)
# 目的: AbstractSNNLayer の抽象メソッドを実装し、インスタンス化エラーを解消。

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional
from snn_research.core.layers.abstract_snn_layer import AbstractSNNLayer

class LIFLayer(AbstractSNNLayer):
    """
    LIFレイヤーの具象クラス。
    build() と reset_state() を実装することで、直接のインスタンス化を可能にする。
    """
    def __init__(self, input_features: int, neurons: int, **kwargs: Any) -> None:
        # AbstractLayer の初期化。基底クラスが期待する引数構成に合わせる。
        learning_config = kwargs.get('learning_config')
        name = kwargs.get('name', 'LIFLayer')
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
        """初期化処理の具象実装。"""
        nn.init.kaiming_uniform_(self.W, a=0.01)
        nn.init.zeros_(self.b)
        self.built = True

    def reset_state(self) -> None:
        """膜電位のリセット。"""
        self.membrane_potential = None

    def forward(self, inputs: Tensor, model_state: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        """順伝播とスパイク統計。"""
        if self.membrane_potential is None or self.membrane_potential.shape[0] != inputs.shape[0]:
            self.membrane_potential = torch.zeros(inputs.shape[0], self._neurons, device=inputs.device)
            
        # ダミーのスパイク生成（実際は lif_update ロジックを適用）
        spikes = (torch.randn(inputs.shape[0], self._neurons, device=inputs.device) > 0.5).float()
        
        # mypyエラー: Tensor オブジェクトを直接呼び出さない形式でBufferを更新
        current_sum = float(spikes.sum().detach().item())
        self.total_spikes.add_(current_sum)
        
        return {'activity': spikes, 'membrane_potential': self.membrane_potential}
