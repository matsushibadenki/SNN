# snn_research/core/networks/bio_pc_network.py
# 生物学的な予測符号化（Bio-PC）を実現するためのネットワーククラス
#
# ディレクトリ: snn_research/core/networks/bio_pc_network.py
# ファイル名: Bio-PC ネットワーク実装
# 目的: 生成・推論ニューロンを組み合わせ、k-WTA等を用いた予測符号化を行う。
#
# 変更点:
# - [修正 v13] mypy修正: PredictiveCodingLayerの必須引数(neuron_class, neuron_params)を追加。
# - [修正 v13] デフォルトニューロンとして AdaptiveLIFNeuron を指定し整合性を確保。

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any, Union, Callable, cast, Type
from .abstract_snn_network import AbstractSNNNetwork
from ..layers.predictive_coding import PredictiveCodingLayer
from ..neurons.feel_neuron import AdaptiveLIFNeuron # プロジェクト構造に合わせたインポート

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
        
        # ニューロンの設定 (mypy必須引数への対応)
        self.neuron_class = neuron_class or AdaptiveLIFNeuron
        self.neuron_params = neuron_params or {"tau_mem": 20.0, "base_threshold": 1.0}

        self.pc_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            # 実定義に合わせて引数を渡す: d_model, d_state, neuron_class, neuron_params
            layer = PredictiveCodingLayer(
                d_model=layer_sizes[i],
                d_state=layer_sizes[i+1],
                neuron_class=self.neuron_class,
                neuron_params=self.neuron_params,
                sparsity=sparsity
            )
            self.pc_layers.append(layer)

    def reset_state(self) -> None:
        """無限再帰を防ぎつつ状態をリセット。"""
        for m in self.modules():
            if m is self: continue
            reset_func = getattr(m, 'reset_state', None)
            if callable(reset_func):
                try:
                    reset_func()
                except Exception:
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理。
        PredictiveCodingLayer.forward(input, state) の引数構成に合わせる必要があるが、
        ここでは簡易的にボトムアップ入力を伝播させる。
        """
        x = x * self.input_gain
        batch_size = x.size(0)
        current_input = x
        
        # PCレイヤー特有の状態管理（簡易版）
        for layer in self.pc_layers:
            # 内部状態の次元を取得
            d_state = getattr(layer, 'norm_state').normalized_shape[0]
            # 状態を生成（本来は時間ステップごとに更新）
            dummy_state = torch.zeros(batch_size, d_state, device=x.device)
            # 戻り値は (updated_state, error, combined_mem)
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
