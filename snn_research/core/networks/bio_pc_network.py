# snn_research/core/networks/bio_pc_network.py
# 生物学的な予測符号化（Bio-PC）を実現するためのネットワーククラス
#
# ディレクトリ: snn_research/core/networks/bio_pc_network.py
# ファイル名: Bio-PC ネットワーク実装
# 目的: 生成・推論ニューロンを組み合わせ、k-WTA等を用いた予測符号化を行う。
#
# 変更点:
# - [修正 v11] mypy引数エラー解消: PredictiveCodingLayerの真の引数名 (in_dim, out_dim) を使用。
# - [修正 v11] mypy型エラー解消: reset_state における Callable 判定を厳密化。

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
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
            # mypy修正: PredictiveCodingLayer は in_dim/out_dim を引数に取る
            layer = PredictiveCodingLayer(
                in_dim=layer_sizes[i],
                out_dim=layer_sizes[i+1],
                sparsity=sparsity
            )
            self.pc_layers.append(layer)

    def reset_state(self) -> None:
        """無限再帰を防ぎつつ子モジュールの状態をリセット。"""
        for m in self.modules():
            if m is self:
                continue
            # mypy修正: 属性の存在と呼び出し可能性を同時にチェック
            reset_func = getattr(m, 'reset_state', None)
            if reset_func is not None and callable(reset_func):
                try:
                    reset_func()
                except TypeError:
                    # メソッドではなくTensor属性などの場合をスキップ
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.input_gain
        current_input = x
        for layer in self.pc_layers:
            current_input = layer(current_input)
        return current_input

    def get_sparsity_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.get_device())
        for layer in self.pc_layers:
            loss_attr = getattr(layer, 'get_sparsity_loss', 0.0)
            if callable(loss_attr):
                total_loss += loss_attr()
            else:
                # Tensorとしての加算
                total_loss += cast(torch.Tensor, loss_attr)
        return total_loss

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
