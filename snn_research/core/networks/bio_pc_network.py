# snn_research/core/networks/bio_pc_network.py
# 生物学的な予測符号化（Bio-PC）を実現するためのネットワーククラス
#
# ディレクトリ: snn_research/core/networks/bio_pc_network.py
# ファイル名: Bio-PC ネットワーク実装
#
# 変更点:
# - [修正 v8] mypyエラー解消: PredictiveCodingLayerへの引数名を input_size/output_size 等の正しい名称へ修正。

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
from .abstract_snn_network import AbstractSNNNetwork
from ..layers.predictive_coding import PredictiveCodingLayer

class BioPCNetwork(AbstractSNNNetwork):
    def __init__(self, 
                 layer_sizes: List[int], 
                 sparsity: float = 0.05, 
                 input_gain: float = 1.0,
                 **kwargs: Any):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.sparsity = sparsity
        self.input_gain = input_gain

        self.pc_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            # mypy修正: PredictiveCodingLayer の実際の引数名に合わせて修正
            layer = PredictiveCodingLayer(
                input_size=layer_sizes[i],    # in_features -> input_size
                output_size=layer_sizes[i+1],  # out_features -> output_size
                sparsity=sparsity
            )
            self.pc_layers.append(layer)
            
        self.last_activations: Dict[str, torch.Tensor] = {}

    def reset_state(self) -> None:
        for name, m in self.named_modules():
            if m is self: continue
            if hasattr(m, 'reset_state') and callable(getattr(m, 'reset_state')):
                m.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.input_gain
        current_input = x
        for i, layer in enumerate(self.pc_layers):
            current_input = layer(current_input)
        return current_input

    def get_sparsity_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.get_device())
        for layer in self.pc_layers:
            if hasattr(layer, 'calculate_sparsity_loss'):
                total_loss += layer.calculate_sparsity_loss()
        return total_loss

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
