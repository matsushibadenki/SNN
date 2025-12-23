# /snn_research/core/layers/lif_layer.py
# 日本語タイトル: LIF SNNレイヤー (物理ダイナミクス安定版)
# 目的: 正しいLIFモデルの実装と、計算グラフの勾配不連続性を解消する。

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
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        # 高精度なSigmoid導関数近似
        sgax = (input * alpha).sigmoid()
        grad_input = grad_output * (1.0 - sgax) * sgax * alpha
        return grad_input, None

class LIFLayer(AbstractSNNLayer):
    """
    LIF（Leaky Integrate-and-Fire）ニューロンレイヤー。
    バッファ演算の最適化により、分散学習時や型チェック時の安定性を確保。
    """
    def __init__(self, input_features: int, neurons: int, **kwargs: Any) -> None:
        learning_config = kwargs.get('learning_config', {})
        name = kwargs.get('name', 'LIFLayer')
        super().__init__((input_features,), (neurons,), learning_config, name)
        
        self._input_features = input_features
        self._neurons = neurons
        
        # LIFパラメータ
        self.decay = kwargs.get('decay', 0.9)  # 減衰率（時定数に依存）
        self.threshold = kwargs.get('threshold', 1.0)
        self.v_reset = kwargs.get('v_reset', 0.0)
        
        # 重みとバイアス
        self.W = nn.Parameter(torch.empty(neurons, input_features))
        self.b = nn.Parameter(torch.empty(neurons))
        
        self.membrane_potential: Optional[Tensor] = None
        self.surrogate_function = SurrogateHeaviside.apply
        
        # 統計用バッファ（モデル保存・同期対象）
        self.register_buffer('total_spikes', torch.tensor(0.0))
        self.build()

    def build(self) -> None:
        """重みのKaiming初期化。"""
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)
        self.built = True

    def reset_state(self) -> None:
        """膜電位の状態を完全にクリア。"""
        self.membrane_potential = None

    def forward(self, inputs: Tensor, model_state: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        """
        物理ダイナミクスの計算ステップ。
        1. 入力積分 2. 減衰 3. 発火判定 4. リセット
        """
        if not self.built:
            self.build()

        # 線形変換（シナプス入力）
        synaptic_input = torch.nn.functional.linear(inputs, self.W, self.b)

        # 膜電位の状態管理
        if self.membrane_potential is None or self.membrane_potential.shape != synaptic_input.shape:
            self.membrane_potential = torch.zeros_like(synaptic_input)
            
        # --- LIFダイナミクス ---
        # 1. 積分と減衰 (Leaky Integration)
        self.membrane_potential = self.membrane_potential * self.decay + synaptic_input
        
        # 2. 発火判定 (Fire)
        # 閾値を超えた場合にスパイクを生成。代理勾配により誤差逆伝播が可能。
        v_shifted = self.membrane_potential - self.threshold
        spikes = self.surrogate_function(v_shifted)
        
        # 3. リセット (Hard Reset)
        # スパイクが発生した場所の膜電位を v_reset に戻す
        mask = (spikes > 0.0).float()
        self.membrane_potential = self.membrane_potential * (1.0 - mask) + self.v_reset * mask
        
        # 4. 統計更新 (mypy対応: Bufferへの安全な加算)
        current_spikes_sum = spikes.sum().detach()
        # Tensor型であることを明示してインプレース加算
        target_buffer = cast(Tensor, self.total_spikes)
        target_buffer.add_(current_spikes_sum)
        
        return {
            'activity': spikes, 
            'membrane_potential': self.membrane_potential
        }
