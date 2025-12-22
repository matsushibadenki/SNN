# snn_research/core/networks/bio_pc_network.py
# 生物学的な予測符号化（Bio-PC）を実現するためのネットワーククラス
#
# ディレクトリ: snn_research/core/networks/bio_pc_network.py
# ファイル名: Bio-PC ネットワーク実装
# 目的: 生成・推論ニューロンを組み合わせ、バックプロパゲーションを用いない学習を実現する。
#
# 変更点:
# - [修正 v10] mypyエラー解消: PredictiveCodingLayerの引数名を内部実装に合わせ修正。
# - [修正 v10] get_sparsity_loss をメソッドではなく属性(Tensor)として安全に加算。

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any, Union
from .abstract_snn_network import AbstractSNNNetwork
from ..layers.predictive_coding import PredictiveCodingLayer

class BioPCNetwork(AbstractSNNNetwork):
    """予測符号化(PC)の原理に基づいた生物学的ニューラルネットワーク。"""
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
            # mypy修正: 内部定義に合わせる(PredictiveCodingLayerは in_features/out_features を期待)
            # もしエラーが出る場合は、**kwargs でラップして動的に渡す
            layer = PredictiveCodingLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i+1],
                sparsity=sparsity
            )
            self.pc_layers.append(layer)

    def reset_state(self) -> None:
        """無限再帰を防ぎつつ状態をリセット。"""
        for m in self.modules():
            if m is self: continue
            if hasattr(m, 'reset_state') and callable(getattr(m, 'reset_state')):
                m.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """AbstractSNNNetworkの型定義(Tensor)を遵守。"""
        x = x * self.input_gain
        current_input = x
        for layer in self.pc_layers:
            current_input = layer(current_input)
        return current_input

    def get_sparsity_loss(self) -> torch.Tensor:
        """各層のスパース性損失を収集。"""
        total_loss = torch.tensor(0.0, device=self.get_device())
        for layer in self.pc_layers:
            # get_sparsity_loss が属性(Tensor)として定義されている場合を考慮
            loss_val = getattr(layer, 'get_sparsity_loss', 0.0)
            if callable(loss_val):
                total_loss += loss_val()
            else:
                total_loss += loss_val
        return total_loss

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
