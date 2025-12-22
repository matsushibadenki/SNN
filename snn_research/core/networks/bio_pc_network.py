# snn_research/core/networks/bio_pc_network.py
# 生物学的な予測符号化（Bio-PC）を実現するためのネットワーククラス
#
# ディレクトリ: snn_research/core/networks/bio_pc_network.py
# ファイル名: Bio-PC ネットワーク実装
#
# 変更点:
# - [修正 v14] mypy修正: AdaptiveLIFNeuron のインポートパスを core.neurons に変更。
# - [修正 v14] mypy修正: PredictiveCodingLayer への引数を位置引数に統一し call-arg を解消。

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any, Union, Callable, cast, Type

from .abstract_snn_network import AbstractSNNNetwork
from ..layers.predictive_coding import PredictiveCodingLayer
from ..neurons import AdaptiveLIFNeuron  # 正しいインポートパスに変更

class BioPCNetwork(AbstractSNNNetwork):
    """
    予測符号化(PC)の原理に基づいた生物学的ニューラルネットワーク。
    """
    def __init__(self, 
                 layer_sizes: List[int], 
                 sparsity: float = 0.05, 
                 input_gain: float = 1.0,
                 neuron_class: Optional[Type[nn.Module]] = None,
                 neuron_params: Optional[Dict[str, Any]] = None,
                 **kwargs: Any):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.sparsity = sparsity
        self.input_gain = input_gain
        
        self.neuron_class = neuron_class or AdaptiveLIFNeuron
        self.neuron_params = neuron_params or {"tau_mem": 20.0, "base_threshold": 1.0}

        self.pc_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            # mypy修正: PredictiveCodingLayer (d_model, d_state, class, params)
            layer = PredictiveCodingLayer(
                layer_sizes[i],
                layer_sizes[i+1],
                self.neuron_class,
                self.neuron_params,
                sparsity=sparsity
            )
            self.pc_layers.append(layer)

    def reset_state(self) -> None:
        for m in self.modules():
            if m is self: continue
            reset_func = getattr(m, 'reset_state', None)
            if callable(reset_func):
                try:
                    reset_func()
                except Exception:
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.input_gain
        batch_size = x.size(0)
        current_input = x
        
        for layer in self.pc_layers:
            # 状態変数の次元を安全に取得
            norm_layer = getattr(layer, 'norm_state', None)
            d_state = norm_layer.normalized_shape[0] if norm_layer else 256
            
            dummy_state = torch.zeros(batch_size, d_state, device=x.device)
            # (updated_state, error, combined_mem) のうち error を次層へ
            _, current_input, _ = layer(current_input, dummy_state)
            
        return current_input

    def get_sparsity_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.get_device())
        for layer in self.pc_layers:
            loss_attr = getattr(layer, 'get_sparsity_loss', 0.0)
            if callable(loss_attr):
                total_loss += loss_attr()
            else:
                total_loss += cast(torch.Tensor, torch.as_tensor(loss_attr))
        return total_loss

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
