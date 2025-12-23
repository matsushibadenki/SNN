# /snn_research/core/layers/lif_layer.py
# 日本語タイトル: LIF SNNレイヤー (状態管理・統計強化版)
# 目的: 正しいLIFモデルの実装と、膜電位の状態管理・発火統計取得を安定させる。

import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Dict, Any, Optional, Tuple, cast
from snn_research.core.layers.abstract_snn_layer import AbstractSNNLayer

class SurrogateHeaviside(torch.autograd.Function):
    """
    ステップ関数の不連続性を解消するための代理勾配（Surrogate Gradient）。
    """
    @staticmethod
    def forward(ctx: Any, input: Tensor, alpha: float = 2.0) -> Tensor:
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        # Sigmoid導関数による近似: f'(x) = alpha * sigmoid(alpha*x) * (1 - sigmoid(alpha*x))
        sgax = (input * alpha).sigmoid()
        grad_input = grad_output * (1.0 - sgax) * sgax * alpha
        return grad_input, None

class LIFLayer(AbstractSNNLayer):
    """
    LIF（Leaky Integrate-and-Fire）ニューロンレイヤー。
    膜電位の適切なリセットと発火率の統計管理機能を備える。
    """
    def __init__(self, input_features: int, neurons: int, **kwargs: Any) -> None:
        learning_config = kwargs.get('learning_config', {})
        name = kwargs.get('name', 'LIFLayer')
        super().__init__((input_features,), (neurons,), learning_config, name)
        
        self._input_features = input_features
        self._neurons = neurons
        
        # LIFパラメータ
        self.decay = kwargs.get('decay', 0.9)
        self.threshold = kwargs.get('threshold', 1.0)
        self.v_reset = kwargs.get('v_reset', 0.0)
        
        # 重みとバイアス
        self.W = nn.Parameter(torch.empty(neurons, input_features))
        self.b = nn.Parameter(torch.empty(neurons))
        
        # 実行時状態（膜電位）
        self.membrane_potential: Optional[Tensor] = None
        self.surrogate_function = SurrogateHeaviside.apply
        
        # 統計用バッファ
        self.register_buffer('total_spikes', torch.tensor(0.0))
        self.register_buffer('total_steps', torch.tensor(0.0))
        
        self.build()

    def build(self) -> None:
        """重みのKaiming初期化。"""
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)
        self.built = True

    def reset_state(self) -> None:
        """膜電位の状態をリセット（ネットワークを初期状態に戻す）。"""
        self.membrane_potential = None
        # 統計のリセットは通常行わないが、必要に応じて明示的に呼ぶ
        
    def forward(self, inputs: Tensor, model_state: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        """
        LIFダイナミクス計算のメインステップ。
        """
        if not self.built:
            self.build()

        # シナプス入力の計算
        synaptic_input = torch.nn.functional.linear(inputs, self.W, self.b)

        # 膜電位の初期化（バッチサイズに合わせて動的に拡張）
        if self.membrane_potential is None or self.membrane_potential.shape != synaptic_input.shape:
            self.membrane_potential = torch.zeros_like(synaptic_input)
        else:
            # 前のステップの膜電位を再利用（学習時はグラフを維持）
            self.membrane_potential = self.membrane_potential.detach() if not self.training else self.membrane_potential
            
        # 1. Leaky Integration (積分と減衰)
        self.membrane_potential = self.membrane_potential * self.decay + synaptic_input
        
        # 2. Spiking (発火判定)
        v_shifted = self.membrane_potential - self.threshold
        spikes = self.surrogate_function(v_shifted)
        
        # 3. Hard Reset (スパイク発生箇所の電位リセット)
        # 勾配フローを阻害しないように注意
        with torch.no_grad():
            mask = (spikes > 0.0).float()
        self.membrane_potential = self.membrane_potential * (1.0 - mask) + self.v_reset * mask
        
        # 4. 統計更新
        if not self.training:
            self.total_spikes += spikes.sum().detach()
            # バッチサイズ * ニューロン数で割る前の「ステップ回数」を加算
            self.total_steps += float(inputs.size(0))
        
        return {
            'activity': spikes, 
            'membrane_potential': self.membrane_potential
        }

    def get_firing_rate(self) -> float:
        """平均発火率を取得する。"""
        if self.total_steps == 0:
            return 0.0
        # 発火数 / (ステップ合計 * ニューロン数)
        rate = self.total_spikes / (self.total_steps * self._neurons)
        return float(rate.item())
