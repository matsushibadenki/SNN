# ファイルパス: snn_research/core/layers/bit_spike_layer.py
# 日本語タイトル: BitSpike Linear & Conv Layer (1.58bit Quantization / Fixed Compatibility)
# 目的: 重みを {-1, 0, 1} に量子化し、SNNのスパイク入力(0, 1)と合わせて乗算器を不要にする。
#       以前のバージョンとの完全な互換性を維持。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple

def activation_quant(x: torch.Tensor):
    """
    入力の量子化 (8bit近似).
    SNNの場合は入力がスパイク(0/1)であることが多いため、
    この関数は主に非スパイク入力や内部状態の量子化に使用される。
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return y + x - x.detach()

def weight_quant(w: torch.Tensor):
    """
    BitNet b1.58 重み量子化関数:
    Weights -> {-1, 0, +1}
    
    scale = mean(|w|)
    w_quant = round(w / scale) -> clamped to [-1, 1]
    """
    scale = w.abs().mean().clamp(min=1e-5)
    w_scaled = w / scale
    w_quant = w_scaled.round().clamp(-1, 1)
    
    # Straight-Through Estimator (STE)
    # Forward: 量子化された値を使用
    # Backward: 元の勾配をそのまま伝播
    w_quant = (w_quant - w_scaled).detach() + w_scaled
    
    return w_quant, scale

# 互換性のためのエイリアス (テストコード修正用)
bit_quantize_weight = weight_quant

class BitSpikeLinear(nn.Linear):
    """
    BitSpike Linear Layer
    
    通常の nn.Linear を置き換えて使用可能。
    Forward時に重みを動的に量子化する。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, 
                 rms_norm: bool = False, quantize_inference: bool = True):
        # quantize_inference: 互換性維持のための引数（デフォルトTrueで常に量子化）
        super().__init__(in_features, out_features, bias=bias)
        self.rms_norm = rms_norm
        self.quantize_inference = quantize_inference
        
        # 重み初期化の調整 (量子化に適した初期値)
        nn.init.kaiming_uniform_(self.weight, a=2.23) # sqrt(5) approx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Input tensor (Spikes or Analog values)
        """
        # 1. 重みの量子化 {-1, 0, 1}
        w_quant, w_scale = weight_quant(self.weight)
        
        # 2. RMS Norm (オプション: 入力の安定化)
        if self.rms_norm:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

        # 3. 線形変換
        # Phase 2 Optimization: scaleを適用して出力のダイナミックレンジを維持
        out = F.linear(x, w_quant, self.bias)
        
        # スケールを復元 (学習安定性のため)
        out = out * w_scale
        
        return out

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, 1.58bit=True'


class BitSpikeConv2d(nn.Conv2d):
    """
    BitSpike Conv2d Layer (1.58bit Quantized Convolution)
    
    視覚野モデル等で使用される軽量畳み込み層。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]], 
                 stride: Union[int, Tuple[int, ...]] = 1, padding: Union[str, int, Tuple[int, ...]] = 0, 
                 dilation: Union[int, Tuple[int, ...]] = 1, groups: int = 1, bias: bool = True, 
                 padding_mode: str = 'zeros', device=None, dtype=None, quantize_inference: bool = True):
        
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.quantize_inference = quantize_inference
        nn.init.kaiming_uniform_(self.weight, a=2.23)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 重みの量子化
        w_quant, w_scale = weight_quant(self.weight)

        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0) # パディング処理済み
        else:
            padding = self.padding

        # 畳み込み演算
        out = F.conv2d(input, w_quant, self.bias, self.stride,
                       padding, self.dilation, self.groups)
        
        # スケール復元
        out = out * w_scale
        
        return out