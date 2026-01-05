# snn_research/core/networks/bio_pc_network.py
# 生物学的予測符号化（Bio-PC）ネットワーク v2.1
#
# 変更点:
# - 双方向エラー伝播の正式実装: 上位層で発生した予測誤差を下位層の更新に利用する。
# - 推論ループ内で error キャッシュを管理し、層間のフィードバックを実現。

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, cast, Type

from .abstract_snn_network import AbstractSNNNetwork
from ..layers.predictive_coding import PredictiveCodingLayer
from ..neurons import AdaptiveLIFNeuron


class BioPCNetwork(AbstractSNNNetwork):
    """
    予測符号化(PC)の原理に基づいた生物学的ニューラルネットワーク。
    Deep Predictive Coding のアーキテクチャを採用。

    Structure:
        Input <-> [Layer 0] <-> State[1] <-> [Layer 1] <-> State[2] ...

    Flow:
        - Prediction (Top-down): State[i+1] -> Layer[i] -> Prediction of State[i]
        - Error (Bottom-up): State[i] - Prediction -> Error[i]
        - Update: Error[i] updates State[i+1] (Inference Path)
        - Feedback: Error[i+1] (Error of State[i+1]) helps update State[i+1] (Top-down Error)
    """

    def __init__(self,
                 layer_sizes: List[int],
                 sparsity: float = 0.05,
                 input_gain: float = 1.0,
                 inference_steps: int = 8,
                 neuron_class: Optional[Type[nn.Module]] = None,
                 neuron_params: Optional[Dict[str, Any]] = None,
                 **kwargs: Any):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.sparsity = sparsity
        self.input_gain = input_gain
        self.inference_steps = inference_steps

        self.neuron_class = neuron_class or AdaptiveLIFNeuron
        self.neuron_params = neuron_params or {
            "tau_mem": 20.0, "base_threshold": 1.0}

        self.pc_layers = nn.ModuleList()
        # Layer[i] connects State[i] (bottom) and State[i+1] (top)
        for i in range(len(layer_sizes) - 1):
            layer = PredictiveCodingLayer(
                layer_sizes[i],     # Bottom size (Prediction Target)
                layer_sizes[i+1],   # Top size (State Source)
                self.neuron_class,
                self.neuron_params,
                sparsity=sparsity,
                inference_steps=1,  # ネットワークレベルで反復するため、層内は1ステップ
                weight_tying=kwargs.get('weight_tying', True)
            )
            self.pc_layers.append(layer)

    def reset_state(self) -> None:
        for m in self.modules():
            if m is self:
                continue
            reset_func = getattr(m, 'reset_state', None)
            if callable(reset_func):
                try:
                    reset_func()
                except Exception:
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        双方向反復推論プロセス (Bidirectional Iterative Inference)
        """
        x = x * self.input_gain
        batch_size = x.size(0)
        device = x.device

        # 1. 状態変数の初期化
        # states[i] は layer_sizes[i] の次元を持つ
        states = [torch.zeros(batch_size, size, device=device)
                  for size in self.layer_sizes]

        # 最下層 (Sensory Layer) に入力をセット
        states[0] = x

        # 各層の予測誤差を保持するバッファ (Layer[i]が出力するErrorはError[i])
        # Layer[i] connects S[i] and S[i+1], outputs Error of S[i]
        # errors[i] corresponds to the prediction error of state[i]
        errors: List[Optional[torch.Tensor]] = [None] * len(self.pc_layers)

        # 2. 推論ループ (Relaxation)
        for t in range(self.inference_steps):
            new_states = [s.clone() for s in states]
            new_errors = [None] * len(self.pc_layers)

            # 最下層は入力で固定 (Sensory Clamping)
            new_states[0] = x

            # 各レイヤーの更新
            for i, layer in enumerate(self.pc_layers):
                # i番目のレイヤーは State[i] (Bottom) と State[i+1] (Top) を接続
                bottom_val = states[i]
                top_val = states[i+1]

                # 上位層からのエラーフィードバック (Top-Down Error)
                # Layer[i]にとっての上位層エラーは、Layer[i+1]が計算したS[i+1]の予測誤差
                # つまり errors[i+1] が存在すればそれを渡す
                td_error = None
                if i + 1 < len(errors):
                    td_error = errors[i+1]

                # レイヤー計算
                # updated_top: 推論パスによって更新された State[i+1]
                # error_bottom: State[i] の予測誤差
                updated_top, error_bottom, _ = layer(
                    bottom_up_input=bottom_val,
                    top_down_state=top_val,
                    top_down_error=td_error  # 正式実装: 上位エラーを下位へ伝播
                )

                # 更新された状態と計算された誤差を保存
                # Note: S[i+1] は Layer[i] (下からの更新) と Layer[i+1] (上からの更新) の両方から影響を受けるべきだが
                # 現在のアーキテクチャでは Layer[i] が S[i+1] を更新する責務を持つ (Inference Path)
                new_states[i+1] = updated_top
                new_errors[i] = error_bottom

            states = new_states
            errors = new_errors  # type: ignore

        # 最終的な出力は最上位層の状態 (Representation)
        return states[-1]

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
