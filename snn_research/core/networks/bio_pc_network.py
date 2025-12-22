# snn_research/core/networks/bio_pc_network.py
# 生物学的な予測符号化（Bio-PC）を実現するためのネットワーククラス
#
# ディレクトリ: snn_research/core/networks/bio_pc_network.py
# ファイル名: Bio-PC ネットワーク実装
# 目的: 生成ニューロンと推論ニューロンを組み合わせ、k-WTA等を用いた予測符号化を行う。
#
# 変更点:
# - [修正 v7] mypyエラー解消: PredictiveCodingLayerへの引数名を修正。
# - [修正 v7] mypyエラー解消: forwardメソッドの戻り値を親クラス(Tensor)に合わせ、詳細は内部状態として保持。
# - [修正 v7] リセット時の無限再帰防止ロジックを維持。

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any, Union
from .abstract_snn_network import AbstractSNNNetwork
from ..layers.predictive_coding import PredictiveCodingLayer
import logging

logger = logging.getLogger(__name__)

class BioPCNetwork(AbstractSNNNetwork):
    """
    予測符号化(PC)の原理に基づいた生物学的ニューラルネットワーク。
    """
    def __init__(self, 
                 layer_sizes: List[int], 
                 sparsity: float = 0.05, 
                 input_gain: float = 1.0,
                 **kwargs: Any): # trainスクリプトからの余剰引数を受け入れ可能にする
        super().__init__()
        self.layer_sizes = layer_sizes
        self.sparsity = sparsity
        self.input_gain = input_gain

        # レイヤーの構築
        self.pc_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            # mypy修正: 引数名を内部実装に合わせる (in_features -> input_size等、実際の定義に準拠)
            # ここでは一般的な nn.Linear 互換の名称を想定
            layer = PredictiveCodingLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i+1],
                sparsity=sparsity
            )
            self.pc_layers.append(layer)
            
        self.last_activations: Dict[str, torch.Tensor] = {}

    def reset_state(self) -> None:
        """状態リセット（無限再帰防止版）"""
        for name, m in self.named_modules():
            if m is self: continue
            if hasattr(m, 'reset_state') and callable(getattr(m, 'reset_state')):
                m.reset_state()
        self.model_state = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        mypy修正: 戻り値を torch.Tensor のみに限定（AbstractSNNNetworkの制約）。
        詳細な活動データは self.last_activations に格納する。
        """
        x = x * self.input_gain
        current_input = x
        self.last_activations = {"input": x}

        for i, layer in enumerate(self.pc_layers):
            # レイヤーの出力が(output, info_dict)を返す場合
            res = layer(current_input)
            if isinstance(res, tuple):
                current_input, info = res
                for k, v in info.items():
                    self.last_activations[f"layer_{i}_{k}"] = v
            else:
                current_input = res
            self.last_activations[f"layer_{i}"] = current_input

        return current_input

    def get_sparsity_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.get_device())
        for layer in self.pc_layers:
            if hasattr(layer, 'get_sparsity_loss'):
                # mypy修正: 呼び出し可能か確認
                loss_func = getattr(layer, 'get_sparsity_loss')
                if callable(loss_func):
                    total_loss += loss_func()
                else:
                    total_loss += loss_func # Tensor属性の場合
        return total_loss

    def get_device(self) -> torch.device:
        return next(self.parameters()).device
