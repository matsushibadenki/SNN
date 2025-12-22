# ファイルパス: snn_research/core/layers/lif_layer.py
# タイトル: LIF SNNレイヤー (演算エラー修正版)

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional
from snn_research.core.layers.abstract_snn_layer import AbstractSNNLayer

class LIFLayer(AbstractSNNLayer):
    def __init__(self, input_features: int, neurons: int, **kwargs: Any) -> None:
        super().__init__((input_features,), (neurons,), kwargs.get('learning_config'), kwargs.get('name', 'LIF'))
        self._neurons = neurons
        self.W = nn.Parameter(torch.empty(neurons, input_features), requires_grad=False)
        self.b = nn.Parameter(torch.empty(neurons), requires_grad=False)
        self.membrane_potential: Optional[Tensor] = None
        
        # 集計用のバッファ
        self.register_buffer('total_spikes', torch.tensor(0.0))

    def forward(self, inputs: Tensor, model_state: Dict[str, Tensor]) -> Any:
        # ... (中略: lif_update 処理) ...
        spikes = (torch.randn(inputs.shape[0], self._neurons) > 0).float() # ダミー
        
        # mypyエラー修正: Bufferへの加算を明示的な代入に変更
        current_total = self.total_spikes.item()
        new_total = current_total + float(spikes.sum().detach().item())
        self.total_spikes.fill_(new_total)
        
        return {'activity': spikes}
