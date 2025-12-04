# ファイルパス: snn_research/core/networks/sequential_snn_network.py
# Title: シーケンシャルSNNネットワーク (STDP対応)
# Description:
#   nn.Sequentialのように層を直列に接続するが、
#   各層の入出力（プリシナプス/ポストシナプス活動）を model_state に記録する機能を持つ。
#   これにより、STDPなどの局所学習則が各層のローカル情報にアクセス可能になる。

import torch
import torch.nn as nn
from typing import Dict, Any, List, OrderedDict

from .abstract_snn_network import AbstractSNNNetwork
# LIFLayerなどの具体的な層クラスが必要な場合はインポートするが、ここでは汎用的に扱う

class SequentialSNNNetwork(AbstractSNNNetwork):
    """
    層を順番に実行し、各層の入出力を記録するネットワーク。
    """
    def __init__(self, layers: OrderedDict[str, nn.Module]):
        super().__init__()
        self.layers_map = nn.ModuleDict(layers)
        self.layer_order = list(layers.keys())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_input = x
        
        # 入力層の活動として記録 (オプション)
        self.model_state['input_activity'] = x.detach()

        for name in self.layer_order:
            layer = self.layers_map[name]
            
            # 1. プリシナプス活動の記録
            # (重み更新のために、入力xを保持)
            self.model_state[f"pre_activity_{name}"] = current_input.detach()
            
            # 2. 層の実行
            output = layer(current_input)
            
            # 出力がタプル（スパイク, 膜電位）の場合の処理
            if isinstance(output, tuple):
                current_output = output[0]
                # 膜電位なども記録したければここで
                # self.model_state[f"membrane_potential_{name}"] = output[1].detach()
            else:
                current_output = output

            # 3. ポストシナプス活動の記録
            self.model_state[f"post_activity_{name}"] = current_output.detach()
            
            # 次の層へ
            current_input = current_output

        return current_input