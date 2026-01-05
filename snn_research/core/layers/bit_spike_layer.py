# ファイルパス: snn_research/core/layers/bit_spike_layer.py
# Title: BitNet Quantization Layers (Cache Exposure)
# 修正内容: 次元引数のint型キャストを追加し、文字列型の入力によるエラーを防止。

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BitSpikeWeightQuantizer(torch.autograd.Function):
    """
    BitNet b1.58 Weight Quantizer.
    """
    @staticmethod
    def forward(ctx, weight, eps=1e-5):
        gamma = torch.mean(torch.abs(weight)).clamp(min=eps)
        scaled_weight = weight / gamma
        quantized_weight = torch.round(torch.clamp(scaled_weight, -1.0, 1.0))
        return quantized_weight * gamma

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def bit_quantize_weight(weight, eps=1e-5):
    return BitSpikeWeightQuantizer.apply(weight, eps)

class BitSpikeLinear(nn.Linear):
    """
    BitNet Linear Layer.
    """
    def __init__(self, in_features, out_features, bias=True, quantize_inference=True):
        # 修正: 文字列として渡された場合に対応するため int() でキャスト
        in_features = int(in_features)
        out_features = int(out_features)
        super().__init__(in_features, out_features, bias=bias)
        self.quantize_inference = quantize_inference
        self.eps = 1e-5
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Inference Cache Buffer
        self.register_buffer('cached_w_quant', None)
        self.is_cached = False 

    def train(self, mode: bool = True):
        super().train(mode)
        # 学習モードに戻ったらキャッシュを無効化
        if mode:
            self.is_cached = False
            self.cached_w_quant = None

    def forward(self, x):
        # Optimization: Prioritize Inference Path
        if not self.training:
            if self.is_cached and self.cached_w_quant is not None:
                return F.linear(x, self.cached_w_quant, self.bias)
            else:
                # Cache miss: Generate and store
                with torch.no_grad():
                    self.cached_w_quant = bit_quantize_weight(self.weight, self.eps)
                self.is_cached = True
                return F.linear(x, self.cached_w_quant, self.bias)
        
        # Training Path (QAT)
        w_quant = bit_quantize_weight(self.weight, self.eps)
        return F.linear(x, w_quant, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, 1.58bit=True'

class BitSpikeConv2d(nn.Conv2d):
    """
    BitNet Conv2d Layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        # 修正: 型安全性のためにキャスト
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.eps = 1e-5
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
        self.register_buffer('cached_w_quant', None)
        self.is_cached = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.is_cached = False
            self.cached_w_quant = None

    def forward(self, input):
        if not self.training:
            if self.is_cached and self.cached_w_quant is not None:
                w_quant = self.cached_w_quant
            else:
                with torch.no_grad():
                    self.cached_w_quant = bit_quantize_weight(self.weight, self.eps)
                self.is_cached = True
                w_quant = self.cached_w_quant
        else:
            w_quant = bit_quantize_weight(self.weight, self.eps)

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            w_quant, self.bias, self.stride,
                            self._pair(0), self.dilation, self.groups)
        return F.conv2d(input, w_quant, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = super().extra_repr()
        return s + ', 1.58bit=True'