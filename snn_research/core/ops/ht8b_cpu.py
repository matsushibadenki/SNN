# ファイルパス: snn_research/core/ops/ht8b_cpu.py
# タイトル: HT8B (Hybrid Temporal-8-Bit) CPU Implementation
# 目的: BitNet 1.58bit量子化およびスパイクパッキングの純粋なCPU実装
# ポリシー: GPU非依存

import torch
import torch.nn as nn
from typing import Optional


class BitSpikeWeightQuantizer(torch.autograd.Function):
    """
    BitNet b1.58 Weight Quantizer (CPU Optimized).
    Forward: Quantize weights to {-1, 0, +1} * gamma
    Backward: Straight-Through Estimator (STE) - 勾配をそのまま通す
    """
    @staticmethod
    def forward(ctx, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # Gamma (Scaling Factor): mean(abs(W))
        gamma = torch.mean(torch.abs(weight)).clamp(min=eps)

        # Quantization process
        scaled_weight = weight / gamma
        quantized_weight = torch.round(torch.clamp(scaled_weight, -1.0, 1.0))

        # Save gamma for potential use (though not strictly needed for inference)
        ctx.gamma = gamma

        return quantized_weight * gamma

    @staticmethod
    def backward(ctx, grad_output):
        # STE: 勾配は変更せずに通過させる
        return grad_output, None


def bit_quantize_weight(weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    重みを1.58bit化するヘルパー関数。
    """
    return BitSpikeWeightQuantizer.apply(weight, eps)


class HT8BSpikePacker(nn.Module):
    """
    Temporal Spike Packing Module (Float -> Uint8) on CPU
    Compresses temporal dimension by factor of 8.
    Input: (Batch, Time, Features) [0.0 or 1.0]
    Output: (Batch, Time/8, Features) [uint8]
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_dim = x.shape
        if T % 8 != 0:
            raise ValueError(
                f"HT8B: Time dimension ({T}) must be divisible by 8")

        # Reshape for packing: (Batch, PackedTime, 8, Features)
        x_reshaped = x.view(B, T // 8, 8, F_dim)

        # Initialize output tensor
        packed = torch.zeros((B, T // 8, F_dim),
                             dtype=torch.uint8, device=x.device)

        # Bit packing loop (CPU)
        # 8タイムステップ分をビットシフトしながらOR演算で1バイトに詰める
        for i in range(8):
            # スパイクがある(>0.5)なら1、なければ0
            bit = (x_reshaped[:, :, i, :] > 0.5).to(torch.uint8)
            packed |= (bit << i)

        return packed
