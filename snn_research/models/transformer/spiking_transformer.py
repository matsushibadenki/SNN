# ファイルパス: snn_research/models/transformer/spiking_transformer.py
# Title: Spiking Transformer V3 (SCAL Bipolar Attention)
# Description:
#   バイポーラ入力変換を用いたロバストなAttention機構を導入。
#   Q, Kをバイポーラ化(-1/1)してから内積をとることで、ノイズ耐性を向上。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, cast 
import math
import logging

from snn_research.core.base import BaseModel, SNNLayerNorm
from snn_research.core.neurons import AdaptiveLIFNeuron
# BitNetのインポート
try:
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    BitSpikeLinear = nn.Linear # type: ignore

from spikingjelly.activation_based import base as sj_base 
from spikingjelly.activation_based import functional as SJ_F 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BipolarSpikeSelfAttention(nn.Module):
    """
    SCAL (Statistical Centroid Alignment Learning) ベースのAttention。
    QとKをバイポーラ化(-1/1)して内積をとることで、ノイズを相殺する。
    """
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.q_proj = BitSpikeLinear(d_model, d_model)
        self.k_proj = BitSpikeLinear(d_model, d_model)
        self.v_proj = BitSpikeLinear(d_model, d_model)
        self.out_proj = BitSpikeLinear(d_model, d_model)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Linear Projections (Spike-based or Analog)
        q = self.q_proj(x).reshape(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.nhead, self.head_dim).transpose(1, 2)

        # --- SCAL Bipolar Logic ---
        # 入力がスパイク(0/1)に近いと仮定し、-1/1に変換してノイズ中心を0にする
        # Min-Max正規化などを経て、おおよそ0~1の範囲にあるものを想定
        # (x - 0.5) * 2
        q_bipolar = (q - 0.5) * 2.0
        k_bipolar = (k - 0.5) * 2.0
        
        # Bipolar Dot Product Attention
        # noise * noise -> 0 (uncorrelated)
        attn_scores = torch.matmul(q_bipolar, k_bipolar.transpose(-2, -1)) * self.scale
        
        # Softmax (Standard Attention)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Value aggregation
        out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)

class SDSAEncoderLayer(sj_base.MemoryModule):
    """
    SCAL Bipolar Attention + BitNet FFN Encoder Layer.
    """
    neuron_ff: AdaptiveLIFNeuron
    neuron_ff2: AdaptiveLIFNeuron
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, time_steps: int, neuron_config: dict):
        super().__init__()
        # Use SCAL-based Bipolar Attention
        self.attention = BipolarSpikeSelfAttention(d_model, nhead)
        
        self.linear1 = BitSpikeLinear(d_model, dim_feedforward)
        
        lif_params = {k: v for k, v in neuron_config.items() if k in ['tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']}
        self.neuron_ff = cast(AdaptiveLIFNeuron, AdaptiveLIFNeuron(features=dim_feedforward, **lif_params))

        self.linear2 = BitSpikeLinear(dim_feedforward, d_model)
        self.neuron_ff2 = cast(AdaptiveLIFNeuron, AdaptiveLIFNeuron(features=d_model, **lif_params))

        self.norm1 = SNNLayerNorm(d_model)
        self.norm2 = SNNLayerNorm(d_model)

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        self.neuron_ff.set_stateful(stateful)
        self.neuron_ff2.set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        self.neuron_ff.reset()
        self.neuron_ff2.reset()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # 1. Bipolar Attention
        attn_output = self.attention(src)

        # 2. Add & Norm
        x = src + attn_output 
        x = self.norm1(x)

        # 3. BitNet FFN
        ff_spikes, _ = self.neuron_ff(self.linear1(x))
        ff_output_analog = self.linear2(ff_spikes)
        ff_output_spikes, _ = self.neuron_ff2(ff_output_analog)

        # 4. Add & Norm
        x = x + ff_output_spikes
        x = self.norm2(x)

        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class SpikingTransformerV2(BaseModel):
    """
    SCAL対応 Spiking Transformer。
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 nhead: int, 
                 num_encoder_layers: int, 
                 dim_feedforward: int, 
                 time_steps: int, 
                 neuron_config: Dict[str, Any],
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 **kwargs: Any):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embedding.num_patches
        
        self.pos_encoder_text = nn.Parameter(torch.zeros(1, 1024, d_model))
        self.pos_encoder_image = nn.Parameter(torch.zeros(1, num_patches, d_model))
        
        self.layers = nn.ModuleList([
            SDSAEncoderLayer(d_model, nhead, dim_feedforward, time_steps, neuron_config)
            for _ in range(num_encoder_layers)
        ])
        self.norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()
        logging.info("✅ SpikingTransformerV3 (SCAL Bipolar Attention) initialized.")

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                input_images: Optional[torch.Tensor] = None,
                context_embeds: Optional[torch.Tensor] = None,
                return_spikes: bool = False, 
                output_hidden_states: bool = False, 
                **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B: int
        N: int
        x: torch.Tensor
        device: torch.device
        
        SJ_F.reset_net(self)

        if input_ids is not None:
            B, N = input_ids.shape
            device = input_ids.device
            x = self.token_embedding(input_ids)
            # Position encoding logic...
            if N <= self.pos_encoder_text.shape[1]:
                 x = x + self.pos_encoder_text[:, :N, :]
            else:
                 x = x + self.pos_encoder_text[:, :self.pos_encoder_text.shape[1], :]
        
        elif input_images is not None:
            device = input_images.device
            x = self.patch_embedding(input_images)
            B, N, _ = x.shape
            x = x + self.pos_encoder_image
        else:
            raise ValueError("Either input_ids or input_images must be provided.")

        if context_embeds is not None:
            if context_embeds.shape[0] != B:
                 if context_embeds.shape[0] == 1:
                      context_embeds = context_embeds.expand(B, -1, -1)
            if context_embeds is not None:
                x = torch.cat([context_embeds, x], dim=1) 

        outputs_over_time = []
        for layer_module in self.layers:
             layer = cast(SDSAEncoderLayer, layer_module)
             layer.set_stateful(True)

        for t in range(self.time_steps):
            x_step = x 
            for layer_module in self.layers:
                layer = cast(SDSAEncoderLayer, layer_module)
                x_step = layer(x_step) 
            outputs_over_time.append(x_step)

        x_final = torch.stack(outputs_over_time).mean(dim=0)

        avg_spikes_val = 0.0
        if return_spikes:
            avg_spikes_val = self.get_total_spikes() / (B * N * self.time_steps)

        for layer_module in self.layers:
             layer = cast(SDSAEncoderLayer, layer_module)
             layer.set_stateful(False)

        x_final = self.norm(x_final)

        if output_hidden_states:
             output = x_final
        else:
            if input_images is not None and context_embeds is None:
                pooled_output = x_final.mean(dim=1)
                output = self.output_projection(pooled_output)
            else:
                output = self.output_projection(x_final)

        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return output, avg_spikes, mem