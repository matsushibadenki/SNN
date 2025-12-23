# ファイルパス: snn_research/core/layers/bit_spike_layer.py
# Title: BitNet Quantization Layers (Linear & Conv2d)
# Description:
#   BitNet b1.58 に基づく量子化層。
#   重みを {-1, 0, 1} に量子化し、推論時の乗算を排除する。
#   Update: BitSpikeConv2d を追加。

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BitSpikeWeightQuantizer(torch.autograd.Function):
    """
    BitNet b1.58 Weight Quantizer (AbsMean).
    Forward: Weights -> {-1, 0, 1} * gamma
    Backward: Straight-Through Estimator (STE)
    """
    @staticmethod
    def forward(ctx, weight, eps=1e-5):
        # Scale factor gamma = mean(|W|)
        gamma = torch.mean(torch.abs(weight)).clamp(min=eps)
        
        # Quantize
        scaled_weight = weight / gamma
        quantized_weight = torch.round(torch.clamp(scaled_weight, -1.0, 1.0))
        
        # De-scale for consistency
        return quantized_weight * gamma

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Pass gradient through unchanged
        return grad_output, None

def bit_quantize_weight(weight, eps=1e-5):
    return BitSpikeWeightQuantizer.apply(weight, eps)

class BitSpikeLinear(nn.Linear):
    """
    BitNet Linear Layer.
    """
    def __init__(self, in_features, out_features, bias=True, quantize_inference=True):
        super().__init__(in_features, out_features, bias=bias)
        self.quantize_inference = quantize_inference
        self.eps = 1e-5
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # Always quantize weights during training (QAT)
        w_quant = bit_quantize_weight(self.weight, self.eps)
        return F.linear(x, w_quant, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, 1.58bit=True'

class BitSpikeConv2d(nn.Conv2d):
    """
    BitNet Convolutional Layer.
    視覚野などのCNN構造を1.58bit化し、エネルギー効率を向上させる。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.eps = 1e-5
        # 重み初期化は通常のConv2dと同じだが、分散を考慮
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # 重みの量子化
        w_quant = bit_quantize_weight(self.weight, self.eps)
        
        # 量子化重みを用いて畳み込み
        # inputがスパイク(0/1)であれば、実質的に加算のみとなる
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            w_quant, self.bias, self.stride,
                            self._pair(0), self.dilation, self.groups)
        return F.conv2d(input, w_quant, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = super().extra_repr()
        return s + ', 1.58bit=True'
