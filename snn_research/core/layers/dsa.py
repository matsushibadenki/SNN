# ファイルパス: snn_research/core/layers/dsa.py
# Title: SNN-DSA (Causal Masking & Type Fixes)
# 修正内容: is_causal引数の追加、mypy型エラー(LinearLayer)の修正。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

try:
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    BitSpikeLinear = nn.Linear # type: ignore

class DSALayer(nn.Module):
    """
    SNN向け動的スパースアテンション (SNN-DSA) with BitNet weights.
    
    Optimizations:
    - Zero-Overhead T=1
    - Causal Masking Support
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        top_k: int = 4, 
        dropout: float = 0.1,
        neuron_params: Optional[Dict[str, Any]] = None,
        use_bitnet: bool = True,
        is_causal: bool = True  # Added for causal masking support
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.top_k = top_k
        self.scale = self.head_dim ** -0.5
        self.use_bitnet = use_bitnet
        self.is_causal = is_causal
        
        # Linear Layer Factory Strategy for Typing
        self.qkv_proj: nn.Module
        self.out_proj: nn.Module
        
        if use_bitnet:
            self.qkv_proj = BitSpikeLinear(d_model, 3 * d_model, quantize_inference=True)
            self.out_proj = BitSpikeLinear(d_model, d_model, quantize_inference=True)
        else:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model)
            self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        if neuron_params is None:
            neuron_params = {}
        
        # State buffers
        self.register_buffer('mem_pot', torch.zeros(1))
        self.register_buffer('decay', torch.tensor(neuron_params.get('decay', 0.9)))
        self.register_buffer('threshold', torch.tensor(neuron_params.get('threshold', 1.0)))
        
        # Explicit type hints for buffers
        self.mem_pot: torch.Tensor
        self.decay: torch.Tensor
        self.threshold: torch.Tensor
        
        # Legacy neuron object
        from snn_research.core.neurons import AdaptiveLIFNeuron
        self.output_neuron = AdaptiveLIFNeuron(features=d_model, **neuron_params)
        
        self.is_state_initialized = False
        
        # Inference Cache
        self.inference_ready = False
        self.register_buffer('inference_w_v', None)
        self.register_buffer('inference_b_v', None)
        self.register_buffer('inference_w_out', None)
        self.register_buffer('inference_b_out', None)
        
        self.inference_w_v: Optional[torch.Tensor]
        self.inference_b_v: Optional[torch.Tensor]
        self.inference_w_out: Optional[torch.Tensor]
        self.inference_b_out: Optional[torch.Tensor]

    def _init_state(self, batch_size, device):
        if not self.is_state_initialized or self.mem_pot.shape[0] != batch_size:
            self.mem_pot = torch.zeros(batch_size, self.d_model, device=device)
            self.is_state_initialized = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.inference_ready = False
            self.inference_w_v = None
            self.inference_b_v = None
            self.inference_w_out = None
            self.inference_b_out = None

    def _get_quantized_or_raw_weight(self, layer):
        weight = layer.weight
        bias = layer.bias
        if self.use_bitnet:
            if hasattr(layer, 'cached_w_quant') and layer.cached_w_quant is not None:
                weight = layer.cached_w_quant
            else:
                device = weight.device
                dummy = torch.zeros(1, layer.in_features, device=device)
                _ = layer(dummy)
                if hasattr(layer, 'cached_w_quant') and layer.cached_w_quant is not None:
                    weight = layer.cached_w_quant
        return weight, bias

    def _prepare_inference_weights(self):
        if self.inference_ready:
            return
        with torch.no_grad():
            w_qkv, b_qkv = self._get_quantized_or_raw_weight(self.qkv_proj)
            start_idx = 2 * self.d_model
            end_idx = 3 * self.d_model
            self.inference_w_v = w_qkv[start_idx:end_idx, :].contiguous()
            if b_qkv is not None:
                self.inference_b_v = b_qkv[start_idx:end_idx].contiguous()
            else:
                self.inference_b_v = None
            
            w_out, b_out = self._get_quantized_or_raw_weight(self.out_proj)
            self.inference_w_out = w_out.contiguous()
            if b_out is not None:
                self.inference_b_out = b_out.contiguous()
            else:
                self.inference_b_out = None
            self.inference_ready = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        
        # --- T=1 Optimized Path (Real-time Inference) ---
        if T == 1:
            if not self.inference_ready:
                self._prepare_inference_weights()
            
            x_flat = x.squeeze(1)
            
            # Assertions for type checker
            assert self.inference_w_v is not None
            assert self.inference_w_out is not None
            
            # V Projection
            v = F.linear(x_flat, self.inference_w_v, self.inference_b_v)
            
            attn_probs = torch.ones(B, self.num_heads, 1, 1, device=x.device, dtype=x.dtype)
            
            analog_out = F.linear(v, self.inference_w_out, self.inference_b_out)
            analog_out = self.norm(analog_out)
            
            if not self.is_state_initialized:
                self._init_state(B, x.device)
            
            self.mem_pot = self.mem_pot * self.decay + analog_out
            spike = (self.mem_pot > self.threshold).float()
            self.mem_pot = self.mem_pot - spike * self.threshold
            return spike.unsqueeze(1), attn_probs

        # --- T > 1 Path (Batch Processing) ---
        if not self.output_neuron.stateful:
            self.output_neuron.reset()
            self.mem_pot = torch.zeros(B, self.d_model, device=x.device)
            self.is_state_initialized = True

        qkv = self.qkv_proj(x)
        query, key, value = qkv.chunk(3, dim=-1)
        
        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Causal Masking
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn_scores.masked_fill_(causal_mask, float('-inf'))

        # Top-K Masking
        if T > self.top_k:
            topk_vals, topk_indices = torch.topk(attn_scores, self.top_k, dim=-1)
            mask = torch.full(attn_scores.shape, float('-inf'), device=attn_scores.device, dtype=attn_scores.dtype)
            mask.scatter_(dim=-1, index=topk_indices, src=topk_vals)
            attn_scores = mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).reshape(B, T, self.d_model)
        
        analog_out = self.out_proj(context)
        analog_out = self.norm(analog_out)
        
        out_spikes_list = []
        neuron_step = self.output_neuron
        for t in range(T):
            step_spike, _ = neuron_step(analog_out[:, t, :])
            out_spikes_list.append(step_spike)
            
        out_spikes = torch.stack(out_spikes_list, dim=1)
        
        return out_spikes, attn_probs

    def reset_state(self):
        self.output_neuron.reset()
        if self.is_state_initialized:
            self.mem_pot.fill_(0)

DynamicSparseAttention = DSALayer