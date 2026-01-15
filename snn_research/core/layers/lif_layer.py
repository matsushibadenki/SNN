# snn_research/core/layers/lif_layer.py
# ファイルパス: snn_research/core/layers/lif_layer.py
# 日本語タイトル: LIF SNNレイヤー (No-BP / No-MatrixOp版)
# 目的: 誤差逆伝播法と行列演算ライブラリへの依存を排除し、ニューロン個別の挙動として実装。
# 修正: エラーログにある adaptive_bias はこのバージョンには不要なため、
#       確実にこのNo-BP版コードで上書きしてエラーを解消する。

import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Dict, Any, Optional, cast
from snn_research.core.layers.abstract_snn_layer import AbstractSNNLayer


class LIFLayer(AbstractSNNLayer):
    """
    LIF（Leaky Integrate-and-Fire）ニューロンレイヤー。
    ポリシー遵守:
    1. 誤差逆伝播法（Backprop）を使用しない (No Surrogate Gradient)。
    2. 行列演算（matmul/linear）を使用しない (Element-wise / Loop)。
    """

    def __init__(self, input_features: int, neurons: int, **kwargs: Any) -> None:
        learning_config = kwargs.get('learning_config', {})
        name = kwargs.get('name', 'LIFLayer')
        super().__init__((input_features,), (neurons,), learning_config, name)

        self._input_features = input_features
        self._neurons = neurons
        self.decay = kwargs.get('decay', 0.9)
        self.threshold = kwargs.get('threshold', 1.0)
        self.v_reset = kwargs.get('v_reset', 0.0)

        # 重みとバイアス
        self.W = nn.Parameter(torch.empty(neurons, input_features))
        self.b = nn.Parameter(torch.empty(neurons))

        self.membrane_potential: Optional[Tensor] = None

        # 統計用バッファ
        self.register_buffer('total_spikes', torch.tensor(0.0))
        self.register_buffer('total_steps', torch.tensor(0.0))

        self.build()

    def build(self) -> None:
        # 初期化（勾配計算不要）
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)
        self.built = True

    def reset_state(self) -> None:
        self.membrane_potential = None

    def forward(self, inputs: Tensor, model_state: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        """
        順伝播処理。
        行列演算(F.linear)を使用せず、ニューロンごとのシナプス入力を計算する。
        """
        if not self.built:
            self.build()

        # ポリシー: 勾配計算を絶対に行わない
        with torch.no_grad():
            batch_size = inputs.shape[0]

            # シナプス入力の計算 (No Matrix Op implementation)
            # 行列演算の代わりに、ニューロンごとに重み付き和を計算するループ処理
            synaptic_input = torch.zeros(
                batch_size, self._neurons, device=inputs.device)

            # ニューロンごとの処理
            for i in range(self._neurons):
                # i番目のニューロンへの入力 = Σ(入力 * 重み) + バイアス
                # inputs: [Batch, InFeatures]
                # W[i]: [InFeatures]
                weighted_sum = (inputs * self.W[i]).sum(dim=1)
                synaptic_input[:, i] = weighted_sum + self.b[i]

            # 膜電位の更新
            if self.membrane_potential is None or self.membrane_potential.shape != synaptic_input.shape:
                self.membrane_potential = torch.zeros_like(synaptic_input)

            self.membrane_potential = self.membrane_potential * self.decay + synaptic_input

            # 発火判定 (単純なStep関数)
            # 代理勾配は使用しない
            v_shifted = self.membrane_potential - self.threshold
            spikes = (v_shifted > 0.0).float()

            # リセット (Hard Reset)
            self.membrane_potential = self.membrane_potential * \
                (1.0 - spikes) + self.v_reset * spikes

            # 統計更新
            total_spikes = cast(Tensor, self.total_spikes)
            total_steps = cast(Tensor, self.total_steps)
            total_spikes += spikes.sum()
            total_steps += float(batch_size)

        return {
            'activity': spikes,
            'membrane_potential': self.membrane_potential
        }

    def get_firing_rate(self) -> float:
        """平均発火率を取得する。"""
        total_steps = cast(Tensor, self.total_steps)
        total_spikes = cast(Tensor, self.total_spikes)

        if total_steps.item() == 0:
            return 0.0

        rate = total_spikes / (total_steps * self._neurons)
        return float(rate.item())
