# snn_research/core/surrogates.py
# Title: Surrogate Gradient Library (Optimized for Phase 2)
# Description:
#   様々な形状のサロゲート勾配関数を提供するモジュール。
#   Sigmoid, ATan, PiecewiseLeakyReLU, FastSigmoidに加え、
#   学習安定性の高い Triangle (三角波) サロゲートを追加。

import torch
import torch.nn as nn
from typing import Any, Tuple

class SurrogateFunctionBase(torch.autograd.Function):
    """サロゲート勾配の基底クラス"""
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        # Heaviside step function: input > 0 -> 1.0, else 0.0
        return (input > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        raise NotImplementedError

class ATan(SurrogateFunctionBase):
    """ArcTan関数に基づくサロゲート勾配 (Deep Learning向け安定型)"""
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        # derivative of atan(alpha * x) is alpha / (1 + (alpha * x)^2)
        # 分母の数値安定性のために1.0を加算
        grad = alpha / (1.0 + (alpha * input).pow(2))
        return grad_input * grad, None

class Sigmoid(SurrogateFunctionBase):
    """Sigmoid関数に基づくサロゲート勾配 (標準的)"""
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        
        # derivative of sigmoid(alpha * x)
        # sgax = sigmoid(alpha * x)
        sgax = (input * alpha).sigmoid()
        grad = (1.0 - sgax) * sgax * alpha
        return grad_input * grad, None

class PiecewiseLeakyReLU(SurrogateFunctionBase):
    """区分線形関数(矩形近似)に基づくサロゲート勾配 (高速)"""
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha # ここではwidthの逆数として扱う
        grad_input = grad_output.clone()
        
        # |x| < 1/alpha の範囲で勾配 1.0
        mask = (input.abs() < (1.0 / alpha)).float()
        return grad_input * mask, None

class FastSigmoid(SurrogateFunctionBase):
    """
    Fast Sigmoid (Hard Sigmoid近似) に基づくサロゲート勾配
    Phase 2 目標: レイテンシ削減のため、exp/atanを使わず代数演算のみで計算。
    f(x) = x / (1 + |x|) の導関数を使用。
    """
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        
        # derivative of Fast Sigmoid: alpha / (1 + |alpha * x|)^2
        denom = 1.0 + (alpha * input).abs()
        grad = alpha / (denom.pow(2))
        return grad_input * grad, None

class Triangle(SurrogateFunctionBase):
    """
    Triangle (三角波) 近似に基づくサロゲート勾配
    Phase 2 推奨: 深層SNNにおいて勾配消失しにくく、矩形近似より情報の伝達効率が良い。
    導関数は 1 - |alpha * x| (ただし |alpha * x| < 1)
    """
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()

        # grad = max(0, 1 - |alpha * x|) * alpha
        # ReLUを利用して max(0, ...) を表現
        grad = (1.0 - (input * alpha).abs()).relu() * alpha
        
        return grad_input * grad, None

def surrogate_factory(name: str = 'atan', alpha: float = 2.0) -> nn.Module:
    """
    名前からサロゲート関数を生成して返すファクトリ
    
    Args:
        name (str): 'atan', 'sigmoid', 'piecewise', 'fast_sigmoid', 'triangle'
        alpha (float): 勾配の鋭さ (scale factor)
        
    Returns:
        nn.Module: サロゲート適用モジュール
    """
    if name == 'atan':
        class ATanModule(nn.Module):
            def __init__(self, alpha): super().__init__(); self.alpha = alpha
            def forward(self, x): return ATan.apply(x, self.alpha)
        return ATanModule(alpha)
    
    elif name == 'sigmoid':
        class SigmoidModule(nn.Module):
            def __init__(self, alpha): super().__init__(); self.alpha = alpha
            def forward(self, x): return Sigmoid.apply(x, self.alpha)
        return SigmoidModule(alpha)
    
    elif name == 'piecewise':
        class PiecewiseModule(nn.Module):
            def __init__(self, alpha): super().__init__(); self.alpha = alpha
            def forward(self, x): return PiecewiseLeakyReLU.apply(x, self.alpha)
        return PiecewiseModule(alpha)
        
    elif name == 'fast_sigmoid':
        class FastSigmoidModule(nn.Module):
            def __init__(self, alpha): super().__init__(); self.alpha = alpha
            def forward(self, x): return FastSigmoid.apply(x, self.alpha)
        return FastSigmoidModule(alpha)

    elif name == 'triangle':
        class TriangleModule(nn.Module):
            def __init__(self, alpha): super().__init__(); self.alpha = alpha
            def forward(self, x): return Triangle.apply(x, self.alpha)
        return TriangleModule(alpha)
    
    else:
        raise ValueError(f"Unknown surrogate function: {name}")