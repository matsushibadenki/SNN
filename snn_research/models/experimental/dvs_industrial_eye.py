# ファイルパス: snn_research/models/experimental/dvs_industrial_eye.py
# 日本語タイトル: Industrial Eye SNN (Optimized Latency)
# 目的: DVS画像を用いた高速外観検査モデル。
# 修正: ループ内のテンソル操作(permute等)を事前計算し、推論レイテンシを最小化。

import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple

from snn_research.core.layers.dsa import DSALayer
from snn_research.core.neurons import AdaptiveLIFNeuron

logger = logging.getLogger(__name__)

class IndustrialEyeSNN(nn.Module):
    """
    産業用DVS外観検査AI (Latency Optimized).
    GroupNormによる堅牢性を維持しつつ、メモリアクセスを最適化して高速化。
    """
    def __init__(
        self,
        input_resolution: Tuple[int, int] = (128, 128),
        input_channels: int = 2, 
        feature_dim: int = 64,
        num_classes: int = 2, 
        time_steps: int = 4, # Default optimized to 4
        use_dsa: bool = True
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.time_steps = time_steps
        
        # --- Layer 1 ---
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(4, 16) 
        self.lif1 = AdaptiveLIFNeuron(features=16, base_threshold=1.0, tau_mem=10.0)
        
        # --- Layer 2 ---
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.lif2 = AdaptiveLIFNeuron(features=32, base_threshold=1.0, tau_mem=10.0)
        
        # --- Layer 3 (DSA/Projection) ---
        self.use_dsa = use_dsa
        
        h_out = input_resolution[0] // 4
        w_out = input_resolution[1] // 4
        flat_dim = 32 * h_out * w_out
        
        if self.use_dsa:
            self.projection = nn.Linear(flat_dim, feature_dim)
            self.dsa_layer = DSALayer(d_model=feature_dim, num_heads=2)
        else:
            self.projection = nn.Linear(flat_dim, feature_dim)

        # --- Layer 4 (Classifier) ---
        self.lif_out = AdaptiveLIFNeuron(features=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        logger.info(f"👁️ Industrial Eye SNN initialized. Res:{input_resolution}, T:{time_steps}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T, C, H, W = x.shape
        
        # BatchとTimeをマージして一括畳み込み（GPU並列化の最大化）
        x_flat = x.view(B * T, C, H, W)
        
        # --- Layer 1 Processing ---
        # Conv -> Norm
        c1 = self.gn1(self.conv1(x_flat)) 
        H1, W1 = c1.shape[2], c1.shape[3]
        
        # [Optimization] ループ外でメモリレイアウトを最適化
        # (B*T, 16, H1, W1) -> (B, T, 16, H1, W1) -> (B, T, H1, W1, 16) -> (B, T, H1*W1*16)
        # LIFニューロンは (Batch, Features) を期待するため、空間次元をバッチ次元に逃がすかフラット化する
        # ここでは (Batch * Spatial, Channels) の形にして LIF に通すのが効率的
        
        # c1_perm: (B, T, H1, W1, 16)
        c1_perm = c1.view(B, T, 16, H1, W1).permute(0, 1, 3, 4, 2).contiguous()
        
        out_s1_list = []
        # Time Loop (LIF State Update)
        for t in range(T):
            # frame: (B, H1, W1, 16) -> view -> (B*H1*W1, 16)
            frame = c1_perm[:, t].view(-1, 16)
            spk, _ = self.lif1(frame)
            # Restore: (B*H1*W1, 16) -> (B, H1, W1, 16) -> (B, 16, H1, W1)
            # permute back is needed for Conv2d next
            out_s1_list.append(spk.view(B, H1, W1, 16).permute(0, 3, 1, 2))
        
        # Stack: (B, T, 16, H1, W1)
        out_s1 = torch.stack(out_s1_list, dim=1)
        
        # --- Layer 2 Processing ---
        c2 = self.conv2(out_s1.view(B*T, 16, H1, W1))
        c2 = self.gn2(c2)
        H2, W2 = c2.shape[2], c2.shape[3]
        
        # Pre-permute for Layer 2
        c2_perm = c2.view(B, T, 32, H2, W2).permute(0, 1, 3, 4, 2).contiguous()
        
        features_list = []
        total_spikes = 0.0
        
        for t in range(T):
            frame = c2_perm[:, t].view(-1, 32)
            spk, _ = self.lif2(frame)
            
            # Flatten spatial dims for dense layer: (B*H2*W2, 32) -> (B, H2*W2*32)
            spk_flat = spk.view(B, H2*W2, 32).reshape(B, -1)
            
            features_list.append(spk_flat)
            total_spikes += spk.sum().item()
            
        features = torch.stack(features_list, dim=1) # (B, T, FlatDim)
        
        # --- Layer 3 & 4 (Dense & Classifier) ---
        features_proj = self.projection(features) # (B, T, FeatureDim)
        
        if self.use_dsa:
            attn_out, _ = self.dsa_layer(features_proj)
            context = attn_out.mean(dim=1)
        else:
            context = features_proj.mean(dim=1)
            
        spk_out, _ = self.lif_out(context)
        logits = self.classifier(spk_out)
        
        # Stats calculation
        sparsity = 1.0 - (total_spikes / max(1.0, float(features.numel())))
        stats = {
            "sparsity": sparsity,
            "estimated_power_mw": (1.0 - sparsity) * 35.0 
        }
        
        return logits, stats