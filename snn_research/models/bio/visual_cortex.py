# ファイルパス: snn_research/models/bio/visual_cortex.py
# 日本語タイトル: Visual Cortex Model (Phase 2 Robustness / BitNet)
# 目的: 生物学的視覚野を模倣し、側抑制(Lateral Inhibition)によるノイズ耐性と、
#       BitNetによる超低消費電力推論を両立させる。

import torch
import torch.nn as nn
import torch.nn.functional as F
from snn_research.core.layers.lif_layer import LIFLayer
from snn_research.core.layers.bit_spike_layer import BitSpikeConv2d

class LateralInhibition(nn.Module):
    """
    側抑制 (Lateral Inhibition) モジュール
    自身の周囲のニューロンの発火を抑制することで、コントラストを強調し、
    ノイズに対するロバスト性を向上させる (Mexican Hat function 近似)。
    """
    def __init__(self, channels: int, kernel_size: int = 3, inhibition_strength: float = 0.5):
        super().__init__()
        self.channels = channels
        self.strength = inhibition_strength
        self.padding = kernel_size // 2
        
        # 固定の抑制カーネル (学習しない)
        # 中心は0 (自己抑制なし)、周囲は負の値
        self.register_buffer('kernel', torch.zeros(channels, 1, kernel_size, kernel_size))
        
        with torch.no_grad():
            center = kernel_size // 2
            # 単純なBox抑制
            self.kernel.fill_(-1.0)
            self.kernel[:, :, center, center] = 0.0
            # Normalize
            self.kernel /= (kernel_size * kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C, H, W) - Membrane Potential or Spikes
        Output: Inhibited Activity
        """
        if self.strength == 0.0:
            return x
            
        # Grouped Conv2d for channel-wise inhibition
        inhibition = F.conv2d(
            x, 
            self.kernel, 
            padding=self.padding, 
            groups=self.channels
        )
        
        # 元の信号から抑制成分を引く
        return x + self.strength * inhibition


class VisualCortex(nn.Module):
    """
    Phase 2 Optimized Visual Cortex
    - V1/V2 Hierarchy
    - BitSpike Convolution (1.58bit weights)
    - Lateral Inhibition for Robustness
    - Adaptive LIF Neurons
    """
    def __init__(self, in_channels: int = 2, hidden_channels: int = 64):
        super().__init__()
        
        # V1 Area: Edge Detection & Orientation
        self.v1_conv = BitSpikeConv2d(in_channels, hidden_channels, kernel_size=5, stride=2, padding=2)
        self.v1_inhibit = LateralInhibition(hidden_channels, kernel_size=3, inhibition_strength=0.3)
        self.v1_lif = LIFLayer(hidden_channels, hidden_channels) # Dummy sizes for abstract layer compatibility
        # Note: LIFLayer is usually Dense. For Conv, we rely on functional LIF logic or ConvLIF wrapper.
        # Here we implement functional LIF for Conv compatibility within this class.
        
        # V2 Area: Feature Combination
        self.v2_conv = BitSpikeConv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1)
        self.v2_inhibit = LateralInhibition(hidden_channels * 2, kernel_size=3, inhibition_strength=0.2)
        
        # Parameters
        self.v_threshold = 1.0
        self.v_decay = 0.5
        
        # State
        self.mem_v1 = None
        self.mem_v2 = None

    def reset_state(self):
        self.mem_v1 = None
        self.mem_v2 = None

    def forward(self, x: torch.Tensor):
        # x: (Batch, Channel, Height, Width) or (Batch, Time, C, H, W)
        if x.dim() == 5:
            # Time-step processing
            outputs = []
            for t in range(x.shape[1]):
                outputs.append(self._forward_step(x[:, t]))
            return torch.stack(outputs, dim=1)
        else:
            return self._forward_step(x)

    def _forward_step(self, x: torch.Tensor):
        # V1 Processing
        c1 = self.v1_conv(x)
        
        # Initialize membrane
        if self.mem_v1 is None or self.mem_v1.shape != c1.shape:
            self.mem_v1 = torch.zeros_like(c1)
        
        # Integrate & Lateral Inhibition
        # 膜電位に対して側抑制を適用し、過剰な発火を抑える
        self.mem_v1 = self.mem_v1 * self.v_decay + c1
        self.mem_v1 = self.v1_inhibit(self.mem_v1)
        
        # Fire
        spike_v1 = (self.mem_v1 > self.v_threshold).float()
        self.mem_v1 = self.mem_v1 * (1.0 - spike_v1) # Soft Reset could be applied here
        
        # V2 Processing
        c2 = self.v2_conv(spike_v1) # Input is binary spikes (Energy Efficient!)
        
        if self.mem_v2 is None or self.mem_v2.shape != c2.shape:
            self.mem_v2 = torch.zeros_like(c2)
            
        self.mem_v2 = self.mem_v2 * self.v_decay + c2
        self.mem_v2 = self.v2_inhibit(self.mem_v2)
        
        spike_v2 = (self.mem_v2 > self.v_threshold).float()
        self.mem_v2 = self.mem_v2 * (1.0 - spike_v2)
        
        return spike_v2