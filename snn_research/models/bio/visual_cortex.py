# ファイルパス: snn_research/models/bio/visual_cortex.py
# Title: Biomimetic Visual Cortex with BitNet & Top-Down Gating
# Description:
#   V1 -> V2 -> V4 -> IT の階層構造を持つ視覚野モデル。
#   修正: 
#   1. 全てのConv層を BitSpikeConv2d に置換。
#   2. Top-Down Attention (Gating) を追加し、予測符号化的な制御を可能に。

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.layers.bit_spike_layer import BitSpikeConv2d
from snn_research.core.base import SNNLayerNorm

class VisualCortexLayer(nn.Module):
    """
    視覚野の単一層 (例: V1)。
    BitNet Conv -> BatchNorm -> LIF -> Lateral Inhibition
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, neuron_params: Dict[str, Any]):
        super().__init__()
        
        # 1.58bit Conv (乗算フリー)
        self.conv = BitSpikeConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        
        # Batch Norm (SNN向け調整)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # LIF Neuron
        self.neuron = AdaptiveLIFNeuron(features=out_channels, **neuron_params)
        
        # Top-Down Gating用の変調層 (入力チャネル数を合わせるための1x1 Conv)
        # 上位層からのフィードバックを受け取り、発火しやすさを調整する
        self.gate_proj = BitSpikeConv2d(out_channels, out_channels, kernel_size=1)
        self.gate_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, top_down_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Feed-forward processing
        mem_pot = self.bn(self.conv(x))
        
        # Top-Down Modulation
        if top_down_signal is not None:
            # 上位層の信号を現在の層のサイズに合わせる (Up-sampling)
            if top_down_signal.shape[-2:] != mem_pot.shape[-2:]:
                top_down_signal = torch.nn.functional.interpolate(
                    top_down_signal, size=mem_pot.shape[-2:], mode='nearest'
                )
            
            # Gating: 膜電位を増幅/抑制
            # gate = sigmoid(W @ feedback)
            gate = self.gate_sigmoid(self.gate_proj(top_down_signal))
            mem_pot = mem_pot * (1.0 + gate) # 注意が向いている領域を強調
            
        spikes, _ = self.neuron(mem_pot)
        return spikes

class VisualCortex(nn.Module):
    """
    階層的視覚野モデル (V1-V2-V4-IT)。
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32, neuron_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        if neuron_params is None:
            neuron_params = {}
            
        # V1: Edge detection (High resolution, low channels)
        self.v1 = VisualCortexLayer(in_channels, base_channels, kernel_size=5, stride=1, neuron_params=neuron_params)
        
        # V2: Texture/Shape parts (Pooling via stride)
        self.v2 = VisualCortexLayer(base_channels, base_channels*2, kernel_size=3, stride=2, neuron_params=neuron_params)
        
        # V4: Object parts
        self.v4 = VisualCortexLayer(base_channels*2, base_channels*4, kernel_size=3, stride=2, neuron_params=neuron_params)
        
        # IT: Object identity (Global pooling done after this)
        self.it = VisualCortexLayer(base_channels*4, base_channels*8, kernel_size=3, stride=2, neuron_params=neuron_params)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = base_channels * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottom-up pass
        # 本来はPCのように双方向ループすべきだが、ここでは簡易的にFF + Feedbackなしで実装
        # 将来的には BioPCNetwork に組み込んで使用する
        
        s1 = self.v1(x)
        s2 = self.v2(s1)
        s4 = self.v4(s2)
        sit = self.it(s4)
        
        # Global Representation
        out = self.global_pool(sit).flatten(1)
        return out
        
    def forward_with_attention(self, x: torch.Tensor, attention_map: torch.Tensor) -> torch.Tensor:
        """
        トップダウン注意マップを用いた推論
        """
        # Attention map is injected at V4 and IT for object selection
        s1 = self.v1(x)
        s2 = self.v2(s1)
        
        # Attention mapを信号として渡す
        s4 = self.v4(s2, top_down_signal=attention_map)
        sit = self.it(s4, top_down_signal=attention_map)
        
        out = self.global_pool(sit).flatten(1)
        return out
