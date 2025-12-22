# snn_research/core/networks/bio_pc_network.py
# 生物学的な予測符号化（Bio-PC）を実現するためのネットワーククラス
#
# ディレクトリ: snn_research/core/networks/bio_pc_network.py
# ファイル名: Bio-PC ネットワーク実装
# 目的: 生成・推論ニューロンを組み合わせ、k-WTA等を用いた予測符号化を行う。
#
# 変更点:
# - [修正 v12] mypy修正: 引数名の不一致を避けるため位置引数形式を採用。
# - [修正 v12] mypy修正: typing.cast のインポート漏れを解消。
# - [修正 v12] get_sparsity_loss の型安全性を向上。

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any, Union, Callable, cast
from .abstract_snn_network import AbstractSNNNetwork
from ..layers.predictive_coding import PredictiveCodingLayer

class BioPCNetwork(AbstractSNNNetwork):
    """
    予測符号化(PC)の原理に基づいた生物学的ニューラルネットワーク。
    """
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
            # mypy修正: 引数名のエラー(call-arg)を回避するため位置引数を使用
            # layer_sizes[i]: 入力サイズ, layer_sizes[i+1]: 出力サイズ
            layer = PredictiveCodingLayer(
                layer_sizes[i], 
                layer_sizes[i+1],
                sparsity=sparsity
            )
            self.pc_layers.append(layer)

    def reset_state(self) -> None:
        """無限再帰を防ぎつつ子モジュールの状態をリセット。"""
        for m in self.modules():
            if m is self:
                continue
            reset_func = getattr(m, 'reset_state', None)
            if reset_func is not None and callable(reset_func):
                try:
                    reset_func()
                except (TypeError, RecursionError):
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """AbstractSNNNetworkの型定義(Tensor)を遵守。"""
        x = x * self.input_gain
        current_input = x
        for layer in self.pc_layers:
            current_input = layer(current_input)
        return current_input

    def get_sparsity_loss(self) -> torch.Tensor:
        """各層からスパース性損失を収集。"""
        total_loss = torch.tensor(0.0, device=self.get_device())
        for layer in self.pc_layers:
            loss_attr = getattr(layer, 'get_sparsity_loss', 0.0)
            if callable(loss_attr):
                total_loss += loss_attr()
            else:
                total_loss += cast(torch.Tensor, loss_attr)
        return total_loss

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
