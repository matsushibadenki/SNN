# ファイルパス: snn_research/core/layers/bit_spike_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BitSpikeWeightQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        gamma = torch.mean(torch.abs(weight)).clamp(min=1e-5)
        scaled_weight = weight / gamma
        quantized_weight = torch.round(torch.clamp(scaled_weight, -1, 1))
        return quantized_weight * gamma

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def bit_quantize_weight(weight):
    return BitSpikeWeightQuantizer.apply(weight)

class BitSpikeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, quantize_inference=True):
        super().__init__(in_features, out_features, bias=bias)
        self.quantize_inference = quantize_inference
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        w_quant = bit_quantize_weight(self.weight)
        return F.linear(x, w_quant, self.bias)