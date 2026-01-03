# /snn_research/core/layers/lif_layer.py
# 日本語タイトル: LIF SNNレイヤー (mypyエラー解消版)
# 目的: register_bufferに対する演算エラーをcastにより解消し、型安全な統計計算を実現。

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
        sgax = (input * alpha).sigmoid()
        grad_input = grad_output * (1.0 - sgax) * sgax * alpha
        return grad_input, None

class LIFLayer(AbstractSNNLayer):
    """
    LIF（Leaky Integrate-and-Fire）ニューロンレイヤー。
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
        
        self.W = nn.Parameter(torch.empty(neurons, input_features))
        self.b = nn.Parameter(torch.empty(neurons))
        
        self.membrane_potential: Optional[Tensor] = None
        self.surrogate_function = SurrogateHeaviside.apply
        
        # 統計用バッファ
        self.register_buffer('total_spikes', torch.tensor(0.0))
        self.register_buffer('total_steps', torch.tensor(0.0))
        
        self.build()

    def build(self) -> None:
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)
        self.built = True

    def reset_state(self) -> None:
        self.membrane_potential = None

    def forward(self, inputs: Tensor, model_state: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        if not self.built:
            self.build()

        synaptic_input = torch.nn.functional.linear(inputs, self.W, self.b)

        if self.membrane_potential is None or self.membrane_potential.shape != synaptic_input.shape:
            self.membrane_potential = torch.zeros_like(synaptic_input)
        else:
            self.membrane_potential = self.membrane_potential.detach() if not self.training else self.membrane_potential
            
        self.membrane_potential = self.membrane_potential * self.decay + synaptic_input
        
        v_shifted = self.membrane_potential - self.threshold
        spikes = self.surrogate_function(v_shifted)
        
        with torch.no_grad():
            mask = (spikes > 0.0).float()
        self.membrane_potential = self.membrane_potential * (1.0 - mask) + self.v_reset * mask
        
        # [修正] castを使用してTensorとして演算を行う
        if not self.training:
            total_spikes = cast(Tensor, self.total_spikes)
            total_steps = cast(Tensor, self.total_steps)
            total_spikes += spikes.sum().detach()
            total_steps += float(inputs.size(0))
        
        return {
            'activity': spikes, 
            'membrane_potential': self.membrane_potential
        }

    def get_firing_rate(self) -> float:
        """平均発火率を取得する。"""
        # [修正] castを使用して演算エラーを回避
        total_steps = cast(Tensor, self.total_steps)
        total_spikes = cast(Tensor, self.total_spikes)
        
        if total_steps.item() == 0:
            return 0.0
        
        rate = total_spikes / (total_steps * self._neurons)
        return float(rate.item())