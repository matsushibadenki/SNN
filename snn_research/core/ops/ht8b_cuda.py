# ファイルパス: snn_research/core/ops/ht8b_cuda.py
# タイトル: HT8B (Hybrid Temporal-8-Bit) Accelerated Kernels
# 内容: BitNet 1.58bit量子化およびスパイクパッキングを行うFused CUDAカーネルの実装
# 目的: Pythonオーバーヘッドを排除し、推論レイテンシを最小化する

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from typing import Optional

# CUDA Kernel Source Code
# 1.58bit Quantization: x -> {-1, 0, 1}
# HT8B Packing: 8 time-steps boolean -> uint8
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// --------------------------------------------------------
// 1. BitNet 1.58bit Quantization Kernel
// Formula: Round(Clamp(x / gamma, -1, 1)) * gamma
// --------------------------------------------------------

template <typename scalar_t>
__global__ void bitnet_quantize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gamma,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t g = gamma[0]; // Assuming global gamma for simplicity or per-tensor
        scalar_t val = input[idx];
        scalar_t scaled = val / (g + 1e-8);
        
        // Clamp and Round
        scalar_t clamped = scaled > 1.0 ? 1.0 : (scaled < -1.0 ? -1.0 : scaled);
        scalar_t rounded = round(clamped);
        
        output[idx] = rounded * g;
    }
}

torch::Tensor bitnet_quantize_cuda(torch::Tensor input, torch::Tensor gamma) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "bitnet_quantize_cuda", ([&] {
        bitnet_quantize_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            size);
    }));
    
    return output;
}

// --------------------------------------------------------
// 2. HT8B Temporal Spike Packing Kernel
// Compresses 8 timesteps of spikes (float 0/1) into 1 byte (uint8)
// Input: (Batch, Time, Features) -> Output: (Batch, Time/8, Features)
// --------------------------------------------------------

__global__ void ht8b_pack_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    int batch_size,
    int time_steps,
    int features) {
    
    int b = blockIdx.z;
    int f = blockIdx.y * blockDim.y + threadIdx.y;
    int t_out = blockIdx.x * blockDim.x + threadIdx.x; // Index in packed time
    
    int packed_time = time_steps / 8;
    
    if (b < batch_size && f < features && t_out < packed_time) {
        uint8_t byte_val = 0;
        
        // Pack 8 bits
        for (int i = 0; i < 8; ++i) {
            int t_in = t_out * 8 + i;
            int input_idx = b * (time_steps * features) + t_in * features + f;
            
            if (input[input_idx] > 0.5f) { // Threshold check
                byte_val |= (1 << i);
            }
        }
        
        int output_idx = b * (packed_time * features) + t_out * features + f;
        output[output_idx] = byte_val;
    }
}

torch::Tensor ht8b_pack_cuda(torch::Tensor input) {
    // Input shape: (Batch, Time, Features)
    // Time must be divisible by 8
    int batch = input.size(0);
    int time = input.size(1);
    int features = input.size(2);
    
    TORCH_CHECK(time % 8 == 0, "HT8B: Time dimension must be divisible by 8");
    
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    auto output = torch::empty({batch, time / 8, features}, options);
    
    dim3 threads(1, 256, 1);
    dim3 blocks(
        (time / 8 + threads.x - 1) / threads.x,
        (features + threads.y - 1) / threads.y,
        batch
    );
    
    ht8b_pack_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        batch, time, features
    );
    
    return output;
}

"""

cpp_source = r"""
torch::Tensor bitnet_quantize_cuda(torch::Tensor input, torch::Tensor gamma);
torch::Tensor ht8b_pack_cuda(torch::Tensor input);
"""

# Compile and Load CUDA Extension
try:
    ht8b_ops = load_inline(
        name='ht8b_ops',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['bitnet_quantize_cuda', 'ht8b_pack_cuda'],
        verbose=False
    )
    IS_CUDA_AVAILABLE = True
except Exception as e:
    print(f"HT8B Warning: CUDA compilation failed. Falling back to CPU/Python. {e}")
    IS_CUDA_AVAILABLE = False


class HT8BQuantizer(torch.autograd.Function):
    """
    HT8B (BitNet 1.58bit) Quantization Function with CUDA Backend
    """
    @staticmethod
    def forward(ctx, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # Calculate Gamma (Scaling Factor)
        # Gamma is typically mean(abs(W))
        gamma = torch.mean(torch.abs(weight)).clamp(min=eps)
        
        # Save for backward
        ctx.save_for_backward(weight)
        ctx.gamma = gamma

        if weight.is_cuda and IS_CUDA_AVAILABLE:
            # Use Fused CUDA Kernel
            # We pass gamma as a tensor to the kernel
            gamma_t = gamma.unsqueeze(0) if gamma.dim() == 0 else gamma
            return ht8b_ops.bitnet_quantize_cuda(weight, gamma_t)
        else:
            # Fallback Python Implementation
            scaled_weight = weight / gamma
            quantized_weight = torch.round(torch.clamp(scaled_weight, -1.0, 1.0))
            return quantized_weight * gamma

    @staticmethod
    def backward(ctx, grad_output):
        # STE (Straight-Through Estimator)
        # Gradient passes through unchanged
        return grad_output, None

def ht8b_quantize(weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return HT8BQuantizer.apply(weight, eps)


class HT8BSpikePacker(nn.Module):
    """
    Temporal Spike Packing Module (Float -> Uint8)
    Compresses temporal dimension by factor of 8.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Time, Features) Float Tensor [0.0, 1.0]
        Returns:
            packed: (Batch, Time/8, Features) Uint8 Tensor
        """
        if x.is_cuda and IS_CUDA_AVAILABLE:
            return ht8b_ops.ht8b_pack_cuda(x)
        else:
            # Python fallback (slow)
            B, T, F_dim = x.shape
            assert T % 8 == 0, "Time must be divisible by 8"
            x_reshaped = x.view(B, T // 8, 8, F_dim)
            packed = torch.zeros((B, T // 8, F_dim), dtype=torch.uint8, device=x.device)
            
            for i in range(8):
                # Bit shift and OR
                bit = (x_reshaped[:, :, i, :] > 0.5).to(torch.uint8)
                packed |= (bit << i)
            
            return packed

class HT8BLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear with HT8B Acceleration
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.eps = 1e-5
        
    def forward(self, input):
        # Quantize weights on-the-fly (or use cached in inference mode)
        # For simplicity, we do on-the-fly here, but in real deployment,
        # we would cache the integer weights.
        w_quant = ht8b_quantize(self.weight, self.eps)
        return F.linear(input, w_quant, self.bias)