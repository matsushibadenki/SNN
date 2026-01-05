# ファイルパス: snn_research/architectures/spiking_diffusion_model.py
# (修正: name-defined エラー解消、Stub解消)

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Type, Optional, List, cast
import logging

from snn_research.core.base import BaseModel, SNNLayerNorm
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# --- 1. TSM (Temporal-wise Spiking Mechanism) ---
class TemporalSpikingMechanism(nn.Module):
    """
    TSM: 拡散ステップ 't' に応じて入力 'x' をスパイクに変換するエンコーダ。
    """
    neuron: nn.Module

    def __init__(self, in_channels: int, time_steps: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.time_embed = nn.Embedding(1000, in_channels) 
        self.merge_layer = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.neuron = neuron_class(features=in_channels, **neuron_params)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t_emb: torch.Tensor = self.time_embed(t)
        t_emb_spatial: torch.Tensor = t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        merged_input: torch.Tensor = torch.cat([x, t_emb_spatial], dim=1)
        current_input: torch.Tensor = self.merge_layer(merged_input)

        SJ_F.reset_net(self.neuron)
        neuron_module: nn.Module = cast(nn.Module, self.neuron)
        if hasattr(neuron_module, 'set_stateful'):
            getattr(neuron_module, 'set_stateful')(True)

        spikes_history: List[torch.Tensor] = []
        current_input_flat: torch.Tensor = current_input.permute(0, 2, 3, 1).reshape(-1, C)

        for _ in range(self.time_steps):
            spike_t_flat, _ = self.neuron(current_input_flat)
            spike_t: torch.Tensor = spike_t_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
            spikes_history.append(spike_t)

        if hasattr(neuron_module, 'set_stateful'):
            getattr(neuron_module, 'set_stateful')(False)

        return torch.stack(spikes_history, dim=1)

# --- 2. SNN U-Net ブロック ---
class SpikingConvBlock(nn.Module):
    """
    SNN U-Net用の基本的な Conv + Norm + LIF ブロック
    """
    lif: nn.Module

    def __init__(self, in_channels: int, out_channels: int, time_steps: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        self.time_steps = time_steps
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = SNNLayerNorm(out_channels) 
        self.lif = neuron_class(features=out_channels, **neuron_params)

    def forward(self, x_spikes: torch.Tensor) -> torch.Tensor:
        B, T, C_in, H, W = x_spikes.shape
        x_flat_time: torch.Tensor = x_spikes.reshape(B * T, C_in, H, W)
        
        conv_out: torch.Tensor = self.conv(x_flat_time)
        _, C_out, H_out, W_out = conv_out.shape
        conv_out_flat: torch.Tensor = conv_out.permute(0, 2, 3, 1).reshape(-1, C_out)
        norm_out_flat: torch.Tensor = self.norm(conv_out_flat)
        
        norm_out: torch.Tensor = norm_out_flat.reshape(B, T, H_out, W_out, C_out).permute(0, 1, 4, 2, 3)

        SJ_F.reset_net(self.lif)
        neuron_module: nn.Module = cast(nn.Module, self.lif)
        if hasattr(neuron_module, 'set_stateful'):
            getattr(neuron_module, 'set_stateful')(True)
            
        spikes_history: List[torch.Tensor] = []
        for t_idx in range(T):
            norm_out_t: torch.Tensor = norm_out[:, t_idx, ...]
            norm_out_t_flat: torch.Tensor = norm_out_t.permute(0, 2, 3, 1).reshape(-1, C_out)
            spike_t_flat, _ = self.lif(norm_out_t_flat)
            spike_t: torch.Tensor = spike_t_flat.reshape(B, H_out, W_out, C_out).permute(0, 3, 1, 2)
            spikes_history.append(spike_t)

        if hasattr(neuron_module, 'set_stateful'):
            getattr(neuron_module, 'set_stateful')(False)

        return torch.stack(spikes_history, dim=1)

# --- 3. Spiking Diffusion Model (SNN U-Net) ---
class SpikingDiffusionModel(BaseModel):
    """
    SNN U-Net アーキテクチャ (完全実装)。
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        snn_time_steps: int = 8,
        neuron_config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        if neuron_config is None:
            neuron_config = {'type': 'lif', 'tau_mem': 10.0, 'base_threshold': 1.0}
        
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[nn.Module] = AdaptiveLIFNeuron if neuron_type_str == 'lif' else IzhikevichNeuron

        # 1. TSM
        self.tsm = TemporalSpikingMechanism(in_channels, snn_time_steps, neuron_class, neuron_params)

        # 2. Encoder
        self.down1 = SpikingConvBlock(in_channels, base_channels, snn_time_steps, neuron_class, neuron_params)
        self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.down2 = SpikingConvBlock(base_channels, base_channels * 2, snn_time_steps, neuron_class, neuron_params)
        self.pool2 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # 3. Bottleneck
        self.bottleneck = SpikingConvBlock(base_channels * 2, base_channels * 4, snn_time_steps, neuron_class, neuron_params)

        # 4. Decoder
        self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.up1 = SpikingConvBlock(base_channels * 6, base_channels * 2, snn_time_steps, neuron_class, neuron_params)
        
        self.upsample2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.up2 = SpikingConvBlock(base_channels * 3, base_channels, snn_time_steps, neuron_class, neuron_params)

        # 5. Final Output
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        
        self._init_weights()
        logger.info("✅ SpikingDiffusionModel (Full U-Net) initialized.")

    def forward(
        self, 
        x_noisy: torch.Tensor, 
        t_diffusion: torch.Tensor,
        return_spikes: bool = False, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, C, H, W = x_noisy.shape
        device: torch.device = x_noisy.device
        
        x_spikes = self.tsm(x_noisy, t_diffusion)
        T_snn = x_spikes.shape[1]

        d1 = self.down1(x_spikes)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        b = self.bottleneck(p2)

        u1_up = self.upsample1(b)
        u1_cat = torch.cat([u1_up, d2], dim=2)
        u1 = self.up1(u1_cat)
        
        u2_up = self.upsample2(u1)
        u2_cat = torch.cat([u2_up, d1], dim=2)
        u2 = self.up2(u2_cat)

        snn_out_mean = u2.mean(dim=1)
        predicted_noise = self.final_conv(snn_out_mean)

        avg_spikes_val = self.get_total_spikes() / (B * T_snn) if return_spikes and T_snn > 0 else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return predicted_noise, avg_spikes, mem