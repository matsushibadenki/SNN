# snn_research/core/networks/sequential_snn_network.py
# Title: Sequential SNN Network (Supports Spatiotemporal)
# Description:
#   入出力を記録しつつ、(Time, Batch, ...) の5次元テンソルが流れてきた場合も
#   各層が対応していればそのまま通す設計。

import torch
import torch.nn as nn
from typing import OrderedDict, Dict, Union, Any
from .abstract_snn_network import AbstractSNNNetwork

class SequentialSNNNetwork(AbstractSNNNetwork):
    """
    層を順番に実行するコンテナ。
    Multi-stepモードのデータ (T, B, ...) が流れてきても、
    構成するレイヤーが対応していればそのまま処理可能にする。
    """
    def __init__(self, layers: OrderedDict[str, nn.Module]):
        super().__init__()
        self.layers_map = nn.ModuleDict(layers)
        self.layer_order = list(layers.keys())

    def reset(self):
        """全層の状態リセット"""
        self.model_state = {}
        for layer in self.layers_map.values():
            if hasattr(layer, 'reset'):
                layer.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_input = x
        
        # 入力活動の記録 (デバッグ・可視化用)
        # Time次元がある場合もそのまま記録
        self.model_state['input_activity'] = x.detach() if hasattr(x, 'detach') else x

        for name in self.layer_order:
            layer = self.layers_map[name]
            
            # 1. プリシナプス活動
            self.model_state[f"pre_activity_{name}"] = current_input.detach() if hasattr(current_input, 'detach') else current_input
            
            # 2. 層の実行
            # レイヤーが追加引数(state)を必要とするかチェック
            # 基本的には x だけを渡して、内部でstate管理してもらう設計に移行推奨
            output: Any
            
            # シグネチャ検査またはTry-Exceptで柔軟に対応
            try:
                # 独自のAbstractLayer等は第二引数を受け取る可能性がある
                # ただし、SpikingJellyなどモダンな実装は module(x) で完結させることが多い
                output = layer(current_input)
            except TypeError:
                try:
                    output = layer(current_input, self.model_state)
                except TypeError:
                    # 引数が合わない場合はエラー詳細を出すべきだが、ここでは単純実行
                    output = layer(current_input)
            
            # 3. 出力の正規化 (Dict/Tuple -> Tensor)
            next_input: torch.Tensor
            
            if isinstance(output, dict):
                # 出力が辞書の場合の処理
                if 'activity' in output:
                    next_input = output['activity']
                    # 膜電位などの内部状態があれば保存
                    for k, v in output.items():
                        if k != 'activity' and isinstance(v, torch.Tensor):
                            self.model_state[f"{k}_{name}"] = v.detach()
                else:
                    # 最初の値を採用
                    next_input = list(output.values())[0]

            elif isinstance(output, tuple):
                # (spikes, mem) のようなタプル返しの場合
                next_input = output[0]
                if len(output) > 1 and isinstance(output[1], torch.Tensor):
                    self.model_state[f"membrane_potential_{name}"] = output[1].detach()
            else:
                next_input = output

            # 4. ポストシナプス活動
            self.model_state[f"post_activity_{name}"] = next_input.detach() if hasattr(next_input, 'detach') else next_input
            
            # 次の層へ
            current_input = next_input

        return current_input