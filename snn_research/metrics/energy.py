# ファイルパス: snn_research/metrics/energy.py
# 日本語タイトル: Advanced SNN Energy Meter (Phase 2 / BitNet Optimized)
# 目的: SNNのスパイク活動とBitNetの1.58bit演算によるエネルギー消費を推定する。
#       FP32 MAC vs INT1 AC のコスト差を反映し、"1/100 Energy" 目標の達成度を測定する。

import torch
import torch.nn as nn
from typing import Dict, Any, Union

class EnergyMeter:
    """
    SNNおよびBitNetのエネルギー消費量を推定するクラス。
    基準: 45nm CMOSプロセス (Horowitz et al. 2014 / BitNet b1.58 paper)
    
    Energy Cost (pJ):
    - FP32 MAC (Multiply-Accumulate): 4.6 pJ
    - FP32 AC (Accumulate): 0.9 pJ
    - INT8 MAC: 0.2 pJ
    - INT1/1.58bit AC (Accumulate only): 0.03 pJ (推計値: FP32 ACの約1/30)
    
    スパイク通信コスト (SOP):
    - Spike Operation: 0.9 pJ (Accumulate相当)
    """
    
    # Energy constants (pJ)
    E_MAC_FP32 = 4.6
    E_AC_FP32 = 0.9
    E_AC_INT1 = 0.03  # BitNet Advantage
    E_SOP = 0.9       # Spike Operation (Standard SNN)

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_energy_pj = 0.0
        self.layer_counts = {
            "fp32_mac": 0,
            "int1_ac": 0,
            "sop": 0
        }

    def register_spike_activity(self, spikes: torch.Tensor):
        """スパイク活動によるエネルギー (SOP) を加算"""
        num_spikes = spikes.sum().item()
        energy = num_spikes * self.E_SOP
        self.total_energy_pj += energy
        self.layer_counts["sop"] += num_spikes

    def register_layer_compute(self, module: nn.Module, input_shape: torch.Size, output_shape: torch.Size):
        """
        レイヤーごとの演算コストを推定して加算
        BitSpikeLayerの場合は INT1 AC として計算する。
        """
        # バッチサイズを除く演算回数概算
        # Conv2d: H_out * W_out * C_out * (K*K*C_in)
        # Linear: Out * In
        
        flops = 0.0
        is_bitnet = getattr(module, "quantize_inference", False) or "BitSpike" in module.__class__.__name__
        
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            # output volume * kernel volume
            out_elements = output_shape[1:].numel() # C_out * H * W
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            # groups補正は省略（簡易計算）
            flops = out_elements * kernel_ops
            
        elif isinstance(module, nn.Linear):
            flops = module.in_features * module.out_features
            
        # エネルギー計算
        if is_bitnet:
            # BitNet: 乗算なし、加算のみ (Accumulate)
            # さらに1.58bitなのでINT1相当のコスト
            energy = flops * self.E_AC_INT1
            self.layer_counts["int1_ac"] += flops
        else:
            # Standard: FP32 MAC
            energy = flops * self.E_MAC_FP32
            self.layer_counts["fp32_mac"] += flops
            
        self.total_energy_pj += energy

    def report(self) -> Dict[str, Union[float, str]]:
        """エネルギー効率レポートを作成"""
        total_mj = self.total_energy_pj / 1e9  # pJ -> mJ
        
        # ANN換算 (全てFP32 MACで行った場合の想定コスト)
        total_ops = self.layer_counts["fp32_mac"] + self.layer_counts["int1_ac"] + self.layer_counts["sop"]
        ann_equivalent_pj = total_ops * self.E_MAC_FP32
        
        efficiency_ratio = ann_equivalent_pj / (self.total_energy_pj + 1e-9)
        
        return {
            "Total Energy (mJ)": round(total_mj, 4),
            "SOP Count": int(self.layer_counts["sop"]),
            "BitNet Ops": int(self.layer_counts["int1_ac"]),
            "Efficiency vs FP32 ANN": f"{efficiency_ratio:.2f}x"
        }