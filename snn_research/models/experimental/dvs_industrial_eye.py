# ファイルパス: snn_research/models/experimental/dvs_industrial_eye.py
# 日本語タイトル: Industrial Eye SNN (Robust & Fast)
# 目的: DVS画像を用いた高速外観検査モデル。
# 修正: GroupNormを追加してノイズ耐性を向上。パラメータを軽量化しレイテンシを短縮。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple, List

from snn_research.core.layers.dsa import DSALayer
from snn_research.core.neurons import AdaptiveLIFNeuron

logger = logging.getLogger(__name__)

class IndustrialEyeSNN(nn.Module):
    """
    産業用DVS外観検査AI。
    GroupNormを採用し、照明変化やノイズに対して堅牢な特徴抽出を行う。
    """
    def __init__(
        self,
        input_resolution: Tuple[int, int] = (128, 128),
        input_channels: int = 2, 
        feature_dim: int = 64,
        num_classes: int = 2, 
        time_steps: int = 8,
        use_dsa: bool = True
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.time_steps = time_steps
        
        # 1. Spiking Feature Extractor (DVS Encoder)
        # Layer 1
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(4, 16) # Robustness: GroupNorm
        self.lif1 = AdaptiveLIFNeuron(features=16, v_threshold=1.0, tau_mem=10.0)
        
        # Layer 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 32) # Robustness: GroupNorm
        self.lif2 = AdaptiveLIFNeuron(features=32, v_threshold=1.0, tau_mem=10.0)
        
        # 2. Dynamic Sparse Attention (DSA) or GAP
        self.use_dsa = use_dsa
        
        # Output spatial dims
        h_out = input_resolution[0] // 4
        w_out = input_resolution[1] // 4
        flat_dim = 32 * h_out * w_out
        
        if self.use_dsa:
            self.projection = nn.Linear(flat_dim, feature_dim)
            self.dsa_layer = DSALayer(d_model=feature_dim, num_heads=2) # Headsを減らして高速化
        else:
            self.projection = nn.Linear(flat_dim, feature_dim)

        # 3. Classifier
        self.lif_out = AdaptiveLIFNeuron(features=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        logger.info(f"👁️ Industrial Eye SNN initialized. Res:{input_resolution}, DSA:{use_dsa}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x: (Batch, Time, Channels, Height, Width)
        Returns:
            logits, stats
        """
        B, T, C, H, W = x.shape
        
        # Merge Batch & Time for parallel spatial processing
        x_flat = x.view(B * T, C, H, W)
        
        # --- Layer 1 ---
        # Conv -> GroupNorm
        c1 = self.gn1(self.conv1(x_flat)) 
        
        # Time-step integration for LIF
        H1, W1 = H // 2, W // 2
        c1 = c1.view(B, T, 16, H1, W1)
        
        out_s1_list = []
        for t in range(T):
            # (B, 16, H1, W1) -> Permute -> (B*H1*W1, 16)
            frame = c1[:, t].permute(0, 2, 3, 1).contiguous().view(-1, 16)
            spk, _ = self.lif1(frame)
            # Restore -> (B, 16, H1, W1)
            spk = spk.view(B, H1, W1, 16).permute(0, 3, 1, 2)
            out_s1_list.append(spk)
        
        out_s1 = torch.stack(out_s1_list, dim=1)
        
        # --- Layer 2 ---
        c2 = self.conv2(out_s1.view(B*T, 16, H1, W1))
        c2 = self.gn2(c2)
        
        H2, W2 = H // 4, W // 4
        c2 = c2.view(B, T, 32, H2, W2)
        
        features_list = []
        total_spikes = 0.0
        
        for t in range(T):
            frame = c2[:, t].permute(0, 2, 3, 1).contiguous().view(-1, 32)
            spk, _ = self.lif2(frame)
            
            # Flatten spatial: (B, 32, H2, W2) -> (B, FlatDim)
            spk_flat = spk.view(B, H2, W2, 32).permute(0, 3, 1, 2).reshape(B, -1)
            
            features_list.append(spk_flat)
            total_spikes += spk.sum().item()
            
        features = torch.stack(features_list, dim=1) # (B, T, FlatDim)
        
        # --- Projection & Attention ---
        features_proj = self.projection(features) # (B, T, FeatureDim)
        
        if self.use_dsa:
            attn_out, _ = self.dsa_layer(features_proj)
            # Temporal Aggregation (Mean)
            context = attn_out.mean(dim=1)
        else:
            context = features_proj.mean(dim=1)
            
        # --- Classification ---
        spk_out, _ = self.lif_out(context)
        logits = self.classifier(spk_out)
        
        # Stats
        sparsity = 1.0 - (total_spikes / max(1.0, float(features.numel())))
        stats = {
            "sparsity": sparsity,
            "estimated_power_mw": (1.0 - sparsity) * 40.0 # 軽量化したので係数を調整
        }
        
        return logits, stats
