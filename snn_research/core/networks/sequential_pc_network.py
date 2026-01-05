# ファイルパス: snn_research/core/networks/sequential_pc_network.py
# Title: 予測符号化ネットワーク (PC Network) - 修正完了版
# Description:
#   PredictiveCodingLayer を積層したネットワーク。
#   修正点:
#   - forwardメソッドの状態初期化ロジックにおいて、weight_tying=True (inference_fc is None) の場合に
#     RuntimeErrorが発生するバグを修正。generative_fc.in_features から次元を取得するように変更。
#   - layer_states の型ヒントを明示。

import torch
import torch.nn as nn
from typing import List, cast

from .abstract_snn_network import AbstractSNNNetwork
from snn_research.core.layers.predictive_coding import PredictiveCodingLayer

class SequentialPCNetwork(AbstractSNNNetwork):
    """
    PredictiveCodingLayerを多層に積み重ねたネットワーク。
    BreakthroughSNNの内部ロジックをネットワークとしてカプセル化し、
    学習則へのインターフェースを提供する。
    """
    
    def __init__(self, layers: List[PredictiveCodingLayer]):
        super().__init__()
        # ModuleListとして登録
        self.pc_layers = nn.ModuleList(layers)
        
        # 各層の状態(State)を保持するバッファ
        self.layer_states: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理。
        
        Args:
            x: 入力データ (Batch, Dim)
            
        Returns:
            torch.Tensor: 最終層の出力（状態）
        """
        bottom_up_input = x
        batch_size = x.shape[0]
        
        # --- バッチサイズの整合性チェックと状態初期化 ---
        # 状態が存在しない、または層の数が合わない、またはバッチサイズが一致しない場合は再初期化
        if not self.layer_states or \
           len(self.layer_states) != len(self.pc_layers) or \
           self.layer_states[0].shape[0] != batch_size:
            
            # 既存の状態があれば破棄（明示的なリセット）して新しいリストを作成
            new_states: List[torch.Tensor] = []
            
            for layer in self.pc_layers:
                pc_layer = cast(PredictiveCodingLayer, layer)
                
                # --- 修正箇所: d_state の取得ロジック ---
                d_state: int
                if pc_layer.inference_fc is not None:
                    # 推論用FC層がある場合 (weight_tying=False)
                    d_state = pc_layer.inference_fc.out_features
                else:
                    # 重み共有の場合 (weight_tying=True)、生成用FC層の入力次元が状態次元
                    if pc_layer.generative_fc is None:
                         raise RuntimeError("PredictiveCodingLayer.generative_fc is None. Invalid layer configuration.")
                    d_state = pc_layer.generative_fc.in_features
                
                new_states.append(torch.zeros(batch_size, d_state, device=x.device))
            
            self.layer_states = new_states

        final_output = None
        
        for i, layer_module in enumerate(self.pc_layers):
            layer = cast(PredictiveCodingLayer, layer_module)
            layer_name = f"layer_{i}"
            
            # 1. 前回の状態を取得
            top_down_state = self.layer_states[i]
            
            # 2. Pre-Activity の記録
            # PCにおいて、重み更新(Generative)の入力となるのは「現在の状態(top_down_state)」
            self.model_state[f"pre_activity_{layer_name}"] = top_down_state.detach()
            
            # 3. レイヤー実行
            # updated_state: 次の時刻のための状態
            # prediction_error: この層の予測誤差 (次の層へのボトムアップ入力となる)
            updated_state, prediction_error, _ = layer(bottom_up_input, top_down_state)
            
            # 4. 状態更新
            self.layer_states[i] = updated_state
            
            # 5. Prediction Error の記録 (Post-Error)
            # これが学習則で「ポストシナプス側の誤差信号」として使われる
            self.model_state[f"prediction_error_{layer_name}"] = prediction_error.detach()
            
            # 次の層への入力は、現在の層の予測誤差
            bottom_up_input = prediction_error
            
            # 最終層の状態を出力とする
            final_output = updated_state

        # 最終的な出力
        return final_output if final_output is not None else x

    def reset_state(self) -> None:
        super().reset_state()
        # リセット時は空リストにする
        self.layer_states = []