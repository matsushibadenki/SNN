# ファイルパス: snn_research/core/layers/lif_layer.py
# タイトル: LIF SNNレイヤー (物理モデル実装・mypy修正版)
# 目的: 正しいLeaky Integrate-and-Fireダイナミクスを実装し、Buffer演算エラーを解消する。

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional, Tuple, cast
from snn_research.core.layers.abstract_snn_layer import AbstractSNNLayer

class SurrogateHeaviside(torch.autograd.Function):
    """
    発火関数のための代理勾配（Surrogate Gradient）。
    順伝播ではHeaviside関数、逆伝播ではSigmoid導関数などを使用。
    """
    @staticmethod
    def forward(ctx: Any, input: Tensor, alpha: float = 2.0) -> Tensor:
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        # sigmoidの導関数近似: alpha * sigmoid(x) * (1 - sigmoid(x))
        # ここでは簡易的に矩形近似またはアークタンジェント系を使用可能だが、
        # 安定性の高いsigmoidベースを採用
        sgax = (input * alpha).sigmoid_()
        grad_input = grad_output * (1 - sgax) * sgax * alpha
        return grad_input, None

class LIFLayer(AbstractSNNLayer):
    """
    LIFレイヤーの具象クラス。
    物理的に正しい膜電位ダイナミクスを実装。
    mypyエラー [operator] "Tensor" not callable を防ぐため、Buffer操作を最適化。
    """
    def __init__(self, input_features: int, neurons: int, **kwargs: Any) -> None:
        learning_config = kwargs.get('learning_config', {})
        name = kwargs.get('name', 'LIFLayer')
        # AbstractLayer の初期化
        super().__init__((input_features,), (neurons,), learning_config, name)
        
        self._input_features = input_features
        self._neurons = neurons
        
        # パラメータ: 減衰率(decay)と閾値(threshold)
        # configから取得、なければデフォルト値
        self.decay = kwargs.get('decay', 0.5)
        self.threshold = kwargs.get('threshold', 1.0)
        self.v_reset = kwargs.get('v_reset', 0.0)
        
        self.W = nn.Parameter(torch.empty(neurons, input_features), requires_grad=True) # 重みは学習可能に修正
        self.b = nn.Parameter(torch.empty(neurons), requires_grad=True)
        
        self.membrane_potential: Optional[Tensor] = None
        self.surrogate_function = SurrogateHeaviside.apply
        
        # 集計用のバッファを登録
        self.register_buffer('total_spikes', torch.tensor(0.0))
        self.build()

    def build(self) -> None:
        """パラメータの初期化。"""
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5) if hasattr(math, 'sqrt') else 2.23) 
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)
        self.built = True

    def reset_state(self) -> None:
        """膜電位の状態リセット。"""
        self.membrane_potential = None

    def forward(self, inputs: Tensor, model_state: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        """
        順伝播: Leaky Integrate-and-Fire Dynamics
        V[t] = V[t-1] * decay + X[t] * W + b
        Spike = 1 if V[t] >= threshold else 0
        V[t] = V_reset if Spike else V[t]
        """
        if not self.built:
            self.build()

        # 入力の線形変換 I[t] = X[t]W^T + b
        # inputs shape: [Batch, Input_Features]
        current_input = torch.nn.functional.linear(inputs, self.W, self.b)

        # 膜電位の初期化
        if self.membrane_potential is None or self.membrane_potential.shape != current_input.shape:
            self.membrane_potential = torch.zeros_like(current_input, device=inputs.device)
            
        # LIF ダイナミクス
        # 1. 積分 (Decay & Integrate)
        self.membrane_potential = self.membrane_potential * self.decay + current_input
        
        # 2. 発火判定 (Surrogate Gradient適用)
        # thresholdを引いて0基準にする
        spike_input = self.membrane_potential - self.threshold
        spikes = self.surrogate_function(spike_input)
        
        # 3. リセット (Hard Reset or Soft Reset)
        # ここではHard Reset (V = v_reset) を採用
        mask = (spikes > 0.0).float()
        self.membrane_potential = self.membrane_potential * (1 - mask) + self.v_reset * mask
        
        # --- mypyエラー [operator] 修正 ---
        # Bufferテンソルを直接参照して add_ メソッドを呼び出す
        current_spikes_sum = spikes.sum().detach()
        # self.total_spikes は Tensor であることを明示して演算
        target_buffer = cast(Tensor, self.total_spikes)
        target_buffer.add_(current_spikes_sum)
        
        return {'activity': spikes, 'membrane_potential': self.membrane_potential}

import math
