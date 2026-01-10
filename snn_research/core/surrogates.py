# snn_research/core/surrogates.py
# Title: Surrogate Gradient Library
# Description:
#   様々な形状のサロゲート勾配関数を提供するモジュール。
#   Sigmoid, ATan, PiecewiseLeakyReLUなどを実装し、学習の安定性を向上させる。

import torch
import torch.nn as nn
from typing import Any

class SurrogateFunctionBase(torch.autograd.Function):
    """サロゲート勾配の基底クラス"""
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        raise NotImplementedError

class ATan(SurrogateFunctionBase):
    """ArcTan関数に基づくサロゲート勾配"""
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        # derivative of atan(alpha * x) is alpha / (1 + (alpha * x)^2)
        return grad_input / (1 + (alpha * input).pow(2)) * alpha, None

class Sigmoid(SurrogateFunctionBase):
    """Sigmoid関数に基づくサロゲート勾配"""
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        # derivative of sigmoid(alpha * x)
        sgax = (input * alpha).sigmoid_()
        return grad_input * (1. - sgax) * sgax * alpha, None

class PiecewiseLeakyReLU(SurrogateFunctionBase):
    """区分線形関数(矩形近似)に基づくサロゲート勾配"""
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha # ここではwidthとして扱う
        grad_input = grad_output.clone()
        
        # |x| < 1/alpha の範囲で勾配を通す (rectangular)
        mask = (input.abs() < (1 / alpha)).float()
        return grad_input * mask, None

def surrogate_factory(name: str = 'atan', alpha: float = 2.0) -> nn.Module:
    """
    名前からサロゲート関数を生成して返すファクトリ
    使い方: spike_func = surrogate_factory('sigmoid', alpha=4.0)
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
    
    else:
        raise ValueError(f"Unknown surrogate function: {name}")