# snn_research/models/bio/visual_cortex.py
# ファイルパス: snn_research/models/bio/visual_cortex.py
# 修正内容: mypy型エラー修正（Noneチェック、引数柔軟化）

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.layers.bit_spike_layer import BitSpikeConv2d


class VisualCortexLayer(nn.Module):
    """
    視覚野の単一層 (例: V1)。
    SCAL -> BitNet Conv -> GroupNorm -> LIF
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, neuron_params: Dict[str, Any]):
        super().__init__()

        padding = kernel_size // 2
        self.conv = BitSpikeConv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        # BatchNorm はステートフルなので GroupNorm に変更
        num_groups = 1
        if out_channels >= 8 and out_channels % 8 == 0:
            num_groups = 8
        elif out_channels >= 4 and out_channels % 4 == 0:
            num_groups = 4
        elif out_channels >= 2 and out_channels % 2 == 0:
            num_groups = 2

        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.neuron = AdaptiveLIFNeuron(features=out_channels, **neuron_params)

        self.gate_proj = BitSpikeConv2d(
            out_channels, out_channels, kernel_size=1)
        self.gate_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, top_down_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        # SCAL Bipolar Transform
        x_bipolar = (x - 0.5) * 2.0

        # Conv -> Norm
        mem_pot = self.norm(self.conv(x_bipolar))

        if top_down_signal is not None:
            if top_down_signal.shape[-2:] != mem_pot.shape[-2:]:
                top_down_signal = torch.nn.functional.interpolate(
                    top_down_signal, size=mem_pot.shape[-2:], mode='nearest'
                )
            gate = self.gate_sigmoid(self.gate_proj(top_down_signal))
            mem_pot = mem_pot * (1.0 + gate)

        spikes, _ = self.neuron(mem_pot)
        return spikes

    def reset_state(self) -> None:
        if hasattr(self.neuron, 'reset'):
            self.neuron.reset()


class VisualCortex(nn.Module):
    """
    階層的視覚野モデル (V1-V2-V4-IT)。
    """

    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 32,
                 time_steps: int = 16,
                 neuron_params: Optional[Dict[str, Any]] = None,
                 **kwargs: Any):
        # mypy対策: 予期せぬ引数 (base_channelsなど) をkwargsで吸収
        super().__init__()
        if neuron_params is None:
            neuron_params = {}

        self.time_steps = time_steps
        self.out_dim = base_channels * 8

        self.v1 = VisualCortexLayer(
            in_channels, base_channels, kernel_size=5, stride=1, neuron_params=neuron_params)
        self.v2 = VisualCortexLayer(
            base_channels, base_channels*2, kernel_size=3, stride=2, neuron_params=neuron_params)
        self.v4 = VisualCortexLayer(
            base_channels*2, base_channels*4, kernel_size=3, stride=2, neuron_params=neuron_params)
        self.it = VisualCortexLayer(
            base_channels*4, base_channels*8, kernel_size=3, stride=2, neuron_params=neuron_params)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def reset_state(self) -> None:
        self.v1.reset_state()
        self.v2.reset_state()
        self.v4.reset_state()
        self.it.reset_state()

    def forward_step(self, x: torch.Tensor, attention_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        s1 = self.v1(x)
        s2 = self.v2(s1)
        s4 = self.v4(s2, top_down_signal=attention_map)
        sit = self.it(s4, top_down_signal=attention_map)
        out = self.global_pool(sit).flatten(1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.reset_state()
        outputs = []

        if x.dim() == 4:
            for t in range(self.time_steps):
                out_t = self.forward_step(x)
                outputs.append(out_t)
        elif x.dim() == 5:
            T = x.shape[1]
            for t in range(T):
                out_t = self.forward_step(x[:, t])
                outputs.append(out_t)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        return torch.stack(outputs, dim=1)
