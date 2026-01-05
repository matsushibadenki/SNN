# ファイルパス: snn_research/training/quantization.py
# Title: 量子化モジュール (BitNet & QAT/SpQuant 実装版)
# Description:
# - BitNet b1.58 (The Era of 1-bit LLMs) に基づく BitLinear。
# - 修正: WeightQuantizerで weight_bits 引数を無視していた問題を解消。
#   1.58bit (3値) 以外が指定された場合、一般的な線形量子化を適用するロジックを追加。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any
import torch.quantization

class WeightQuantizer(torch.autograd.Function):
    """
    重みを量子化するカスタム自動微分関数。
    - weight_bits が ~1.58 (log2(3)) の場合: {-1, 0, 1} の3値量子化 (BitNet)。
    - それ以外の場合: 指定ビット数での線形量子化 (k-bit)。
    """
    @staticmethod
    def forward(ctx: Any, weight: torch.Tensor, weight_bits: float) -> torch.Tensor: # type: ignore[override]
        # 1.58bit (BitNet b1.58: Ternary {-1, 0, 1})
        # 浮動小数点の誤差を考慮して 1.58 付近かどうか判定
        if abs(weight_bits - 1.58) < 0.1:
            gamma = torch.mean(torch.abs(weight)).clamp(min=1e-5)
            weight_scaled = weight / gamma
            quantized_weight = torch.round(weight_scaled).clamp(-1, 1)
            return quantized_weight * gamma
        
        # 一般的な k-bit 量子化
        # 符号付き整数 (int8, int4 など) を想定
        # レベル数: 2^k, 範囲: [-2^(k-1), 2^(k-1)-1]
        else:
            num_bits = int(weight_bits)
            if num_bits < 1:
                # 1bit未満はサポート外のため、そのまま返すか1bitにするなどの処理が必要だが
                # ここでは安全のため1.58bitロジックへフォールバック
                gamma = torch.mean(torch.abs(weight)).clamp(min=1e-5)
                return (weight / gamma).round().clamp(-1, 1) * gamma
                
            q_min = -(2 ** (num_bits - 1))
            q_max = (2 ** (num_bits - 1)) - 1
            
            # AbsMax Scaling
            scale = weight.abs().max() / max(abs(q_min), abs(q_max))
            scale = scale.clamp(min=1e-5)
            
            quantized_weight = (weight / scale).round().clamp(q_min, q_max)
            return quantized_weight * scale

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]: # type: ignore[override]
        return grad_output, None

class ActivationQuantizer(torch.autograd.Function):
    """
    活性化値を量子化する関数 (例: 8bit)。
    """
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, num_bits: int = 8) -> torch.Tensor: # type: ignore[override]
        Q_b = 2 ** (num_bits - 1) - 1
        scale = 127.0 / x.abs().max().clamp(min=1e-5)
        y = (x * scale).round().clamp(-Q_b, Q_b) / scale
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]: # type: ignore[override]
        return grad_output, None

class BitLinear(nn.Linear):
    """
    BitNet 1.58bit (or k-bit) Linear Layer.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        weight_bits: float = 1.58,
        quantize_activation: bool = False
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bits = weight_bits
        self.quantize_activation = quantize_activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_quant = WeightQuantizer.apply(self.weight, self.weight_bits)
        if self.quantize_activation and input.is_floating_point():
            input_quant = ActivationQuantizer.apply(input)
        else:
            input_quant = input
        return F.linear(input_quant, w_quant, self.bias)

# ... (apply_qat, convert_to_quantized_model, apply_spquant_quantization は変更なし) ...

def apply_qat(model: nn.Module) -> nn.Module:
    """
    PyTorchのQuantization Aware Training (QAT) 準備を行う。
    """
    backend = 'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else 'qnnpack'
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(model, inplace=True)
    print(f"✅ QAT preparation complete using backend: {backend}")
    return model

def convert_to_quantized_model(model: nn.Module) -> nn.Module:
    """
    QATモデルを推論用の量子化モデルに変換する。
    """
    model.eval()
    torch.quantization.convert(model, inplace=True)
    print("✅ Model converted to quantized version.")
    return model

class SpQuantObserver(nn.Module):
    """SNN膜電位のための量子化オブザーバー"""
    def __init__(self, num_bits: int = 4):
        super().__init__()
        self.num_bits = num_bits
        self.register_buffer('scale', torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            max_val = x.abs().max()
            self.scale = max_val / (2**(self.num_bits - 1) - 1)
        
        x_quant = (x / self.scale).round().clamp(-(2**(self.num_bits-1)), 2**(self.num_bits-1)-1)
        return x_quant * self.scale

def apply_spquant_quantization(model: nn.Module) -> nn.Module:
    """
    SNN特有の膜電位量子化 (SpQuant) を適用する。
    """
    from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
    
    for name, module in model.named_modules():
        if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron)):
            if not hasattr(module, 'quantizer'):
                module.quantizer = SpQuantObserver(num_bits=4) # type: ignore
                
                original_forward = module.forward
                
                def quantized_forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    if hasattr(module, 'quantizer'):
                         x = module.quantizer(x) # type: ignore
                    return original_forward(x)
                
                module.forward = quantized_forward # type: ignore
                print(f"  - Applied SpQuant to {name}")
                
    print("✅ SpQuant applied to neuron layers.")
    return model