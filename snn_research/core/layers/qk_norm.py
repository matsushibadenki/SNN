# Filepath: snn_research/core/layers/qk_norm.py
# Title: Spiking QK Norm (RMSNorm)
# Description: 
#   Implementation of QK-Norm (RMSNorm) for SNN architectures.
#   This corresponds to the "QK-norm SNN custom implementation" in the Phase 3 roadmap.
#   It stabilizes the Query and Key representations before attention.

import torch
import torch.nn as nn

class SpikingQKNorm(nn.Module):
    """
    Spiking QK Norm (RMSNorm) layer.
    Approximates the QK-norm found in Gemma 2/3 for SNN architectures.
    
    This layer normalizes the input tensor using Root Mean Square Normalization.
    It is typically applied to Queries and Keys before the dot product in Attention mechanisms.
    
    Args:
        dim (int): The dimension of the input feature.
        eps (float): Small value to avoid division by zero.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SpikingQKNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Dim) or (Batch, Dim).
                              Can be spikes (0/1) or membrane potentials.
        
        Returns:
            torch.Tensor: Normalized output.
        """
        # x: (B, T, D) or (B, D)
        
        # Ensure calculation in float32 for stability, even if input is fp16/bf16
        input_dtype = x.dtype
        x_f = x.float()
        
        # Calculate RMS: sqrt(mean(x^2))
        mean_square = x_f.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(mean_square + self.eps)
        
        norm_x = x_f * rsqrt
        
        # Apply learnable scale
        output = norm_x * self.scale
        
        return output.to(input_dtype)

    def reset(self):
        """
        Stateless layer, but provided for compatibility with SNN reset_net calls.
        """
        pass
