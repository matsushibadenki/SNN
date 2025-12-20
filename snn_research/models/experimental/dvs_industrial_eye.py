# ファイルパス: snn_research/models/experimental/dvs_industrial_eye.py
# 日本語タイトル: Industrial Eye SNN (DVS Inspection) [Fixed Shapes]
# 目的・内容:
#   ROADMAP v17.0 "Industrial Eye" の実装。
#   修正: ニューロンへの入力形状不一致(RuntimeError)を修正。
#   (Batch, C, H, W) -> (Batch*H*W, C) に変形して LIF を適用するロジックに変更。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple, List

# 依存関係
from snn_research.core.layers.dsa import DSALayer  # Dynamic Sparse Attention
from snn_research.core.neurons import AdaptiveLIFNeuron

logger = logging.getLogger(__name__)

class IndustrialEyeSNN(nn.Module):
    """
    産業用DVS外観検査AI。
    高速ライン上を流れる製品の「動き（Optical Flow）」と「形状（Shape）」を同時に処理し、
    欠陥（傷、異物混入など）を検知する。
    """
    def __init__(
        self,
        input_resolution: Tuple[int, int] = (128, 128),
        input_channels: int = 2, # On/Off polarity
        feature_dim: int = 64,
        num_classes: int = 2, # Normal vs Defect
        time_steps: int = 8,
        use_dsa: bool = True
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.time_steps = time_steps
        
        # 1. Spiking Feature Extractor (DVS Encoder)
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.lif1 = AdaptiveLIFNeuron(features=16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.lif2 = AdaptiveLIFNeuron(features=32)
        
        # 2. Dynamic Sparse Attention (DSA)
        self.use_dsa = use_dsa
        if self.use_dsa:
            # conv2 out: (32, H/4, W/4)
            h_out = input_resolution[0] // 4
            w_out = input_resolution[1] // 4
            flat_dim = 32 * h_out * w_out
            
            self.dsa_layer = DSALayer(d_model=feature_dim, num_heads=4)
            self.projection = nn.Linear(flat_dim, feature_dim)
        else:
            h_out = input_resolution[0] // 4
            w_out = input_resolution[1] // 4
            flat_dim = 32 * h_out * w_out
            self.projection = nn.Linear(flat_dim, feature_dim)

        # 3. Classifier / Anomaly Detector
        self.lif_out = AdaptiveLIFNeuron(features=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        logger.info(f"👁️ Industrial Eye SNN initialized. Resolution: {input_resolution}, DSA: {use_dsa}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x: (Batch, Time, Channels, Height, Width) - DVS Event Frames
        Returns:
            logits: (Batch, NumClasses)
            stats: 統計情報（Sparsity, Energy等）
        """
        B, T, C, H, W = x.shape
        
        # 効率化のため、BatchとTimeをマージしてConvに通す
        x_flat = x.view(B * T, C, H, W)
        
        # --- Layer 1 ---
        c1 = self.conv1(x_flat) # (B*T, 16, H/2, W/2)
        H1, W1 = H // 2, W // 2
        
        # Reshape to (B*T*H1*W1, 16) for LIF neuron (which expects feature dim as last)
        # LIF state will effectively be (B*T*H1*W1, 16) - this resets implicitly if batch size changes?
        # Note: In loop, we treat time step by step.
        
        # 時間展開
        c1 = c1.view(B, T, 16, H1, W1)
        out_s1: List[torch.Tensor] = []
        
        for t in range(T):
            frame = c1[:, t] # (B, 16, H1, W1)
            
            # ConvLIF logic: Permute -> Flatten -> LIF -> Reshape -> Permute
            # (B, H1, W1, 16) -> (B*H1*W1, 16)
            frame_permuted = frame.permute(0, 2, 3, 1).contiguous().view(-1, 16)
            
            spk, _ = self.lif1(frame_permuted) 
            
            # (B*H1*W1, 16) -> (B, H1, W1, 16) -> (B, 16, H1, W1)
            spk = spk.view(B, H1, W1, 16).permute(0, 3, 1, 2)
            out_s1.append(spk)
            
        out_s1_tensor = torch.stack(out_s1, dim=1) # (B, T, 16, H1, W1)
        
        # --- Layer 2 ---
        c2 = self.conv2(out_s1_tensor.view(B * T, 16, H1, W1))
        # (B*T, 32, H/4, W/4)
        H2, W2 = H // 4, W // 4
        c2 = c2.view(B, T, 32, H2, W2)
        
        features_list: List[torch.Tensor] = []
        total_spikes = 0.0
        
        for t in range(T):
            frame = c2[:, t] # (B, 32, H2, W2)
            
            frame_permuted = frame.permute(0, 2, 3, 1).contiguous().view(-1, 32)
            spk, _ = self.lif2(frame_permuted)
            
            # Flatten spatial dimensions for next dense layer
            # (B*H2*W2, 32) -> (B, H2, W2, 32) -> (B, 32, H2, W2) -> (B, 32*H2*W2)
            spk = spk.view(B, H2, W2, 32).permute(0, 3, 1, 2).contiguous().view(B, -1)
            
            features_list.append(spk) 
            total_spikes += spk.sum().item()
            
        # (B, T, FlatDim)
        features = torch.stack(features_list, dim=1)
        
        # --- Layer 3 (Attention / Aggregation) ---
        features_proj = self.projection(features) # (B, T, FeatureDim)
        
        if self.use_dsa:
            attn_out = self.dsa_layer(features_proj) 
            if isinstance(attn_out, tuple): attn_out = attn_out[0]
            
            # 時間方向に集約 (Mean)
            if attn_out.dim() == 3:
                context = attn_out.mean(dim=1)
            else:
                context = attn_out
        else:
            context = features_proj.mean(dim=1)
            
        # --- Layer 4 (Classification) ---
        spk_out, _ = self.lif_out(context)
        logits = self.classifier(spk_out)
        
        # Stats
        sparsity = 1.0 - (total_spikes / max(1.0, float(features.numel())))
        stats = {
            "sparsity": sparsity,
            "mean_firing_rate": (1.0 - sparsity),
            "estimated_power_mw": (1.0 - sparsity) * 50.0 
        }
        
        return logits, stats