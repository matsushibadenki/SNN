# ファイルパス: snn_research/core/networks/sequential_snn_network.py
# Title: シーケンシャルSNNネットワーク (STDP対応・修正版)
# Description:
#   各層の入出力を記録し、局所学習則をサポートするネットワーク。
#   修正点:
#   - forwardメソッドで、各層に model_state を渡すように修正 (AbstractSNNLayer対応)。
#   - 層の出力が辞書型 (Dict[str, Tensor]) の場合、'activity' キーから次の入力を取得するように修正。

import torch
import torch.nn as nn
from typing import OrderedDict

from .abstract_snn_network import AbstractSNNNetwork

class SequentialSNNNetwork(AbstractSNNNetwork):
    """
    層を順番に実行し、各層の入出力を記録するネットワーク。
    AbstractLayer (LIFLayerなど) の積層に対応。
    """
    def __init__(self, layers: OrderedDict[str, nn.Module]):
        super().__init__()
        self.layers_map = nn.ModuleDict(layers)
        self.layer_order = list(layers.keys())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_input = x
        
        # 入力層の活動として記録
        self.model_state['input_activity'] = x.detach()

        for name in self.layer_order:
            layer = self.layers_map[name]
            
            # 1. プリシナプス活動の記録
            self.model_state[f"pre_activity_{name}"] = current_input.detach()
            
            # 2. 層の実行
            # AbstractLayer準拠の場合は model_state を渡す必要がある
            # (簡易的に引数検査を行うか、try-except、あるいは仕様として渡す)
            try:
                output = layer(current_input, self.model_state)
            except TypeError:
                # model_state を受け取らない通常の nn.Module の場合
                output = layer(current_input)
            
            # 3. 出力の処理 (Tensor, Tuple, Dict)
            current_output: torch.Tensor
            
            if isinstance(output, dict):
                # AbstractSNNLayer は {'activity': ..., 'membrane_potential': ...} を返す
                if 'activity' in output:
                    current_output = output['activity']
                    # 膜電位があれば記録
                    if 'membrane_potential' in output:
                        self.model_state[f"membrane_potential_{name}"] = output['membrane_potential'].detach()
                else:
                    # activityキーがない場合は最初の値を採用するなど（状況による）
                    current_output = list(output.values())[0]
            
            elif isinstance(output, tuple):
                current_output = output[0]
                # 2番目の要素が膜電位などの場合
                if len(output) > 1 and isinstance(output[1], torch.Tensor):
                    self.model_state[f"membrane_potential_{name}"] = output[1].detach()
            else:
                current_output = output

            # 4. ポストシナプス活動の記録
            self.model_state[f"post_activity_{name}"] = current_output.detach()
            
            # 次の層へ
            current_input = current_output

        return current_input