# ファイルパス: snn_research/core/networks/sequential_pc_network.py
# Title: 予測符号化ネットワーク (PC Network) - バッチサイズ対応版
# 機能説明:
#   PredictiveCodingLayer を積層したネットワーク。
#   バッチサイズが動的に変化する場合（例：エポックの最後のバッチ）に対応するため、
#   入力バッチサイズと保持している状態のサイズを比較し、不一致があれば状態をリセットする処理を追加。
#
# 修正 (mypy):
#   - layer_states の型を明示的に List[torch.Tensor] とすることで、[has-type] エラーを解消。
#   - inference_fc が None でないことを確認して [union-attr] エラーを解消。

import torch
import torch.nn as nn
from typing import List, Any, Dict, cast, Optional

from .abstract_snn_network import AbstractSNNNetwork
from snn_research.core.layers.predictive_coding import PredictiveCodingLayer

class SequentialPCNetwork(AbstractSNNNetwork):
    """
    PredictiveCodingLayerを多層に積み重ねたネットワーク。
    BreakthroughSNNの内部ロジックをネットワークとしてカプセル化し、
    学習則へのインターフェースを提供する。
    """
    # クラスレベルで型ヒントを与えることも可能だが、__init__内での初期化がベストプラクティス
    # ここでは動的に生成されるため、forward内で管理するが、型アノテーションを追加する

    def __init__(self, layers: List[PredictiveCodingLayer]):
        super().__init__()
        # ModuleListとして登録
        self.pc_layers = nn.ModuleList(layers)
        
        # 各層の状態(State)を保持するバッファ
        # mypyエラー解消のため、明示的に型ヒント付きで初期化
        self.layer_states: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: 入力データ (Batch, Dim)
        """
        bottom_up_input = x
        batch_size = x.shape[0]
        
        # --- バッチサイズの整合性チェックと状態初期化 ---
        # 状態が存在しない、または層の数が合わない、またはバッチサイズが一致しない場合は再初期化
        if not self.layer_states or \
           len(self.layer_states) != len(self.pc_layers) or \
           self.layer_states[0].shape[0] != batch_size:
            
            # 既存の状態があれば破棄（明示的なリセット）して新しいリストを作成
            # ここで型ヒントを明示する
            new_states: List[torch.Tensor] = []
            
            for layer in self.pc_layers:
                pc_layer = cast(PredictiveCodingLayer, layer)
                # 推論ニューロンの出力次元を取得
                # inference_fc は nn.Linear なので out_features を持つ
                # mypy エラー解消: None でないことを確認
                if pc_layer.inference_fc is None:
                    raise RuntimeError("PredictiveCodingLayer.inference_fc is None")
                
                d_state = pc_layer.inference_fc.out_features 
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