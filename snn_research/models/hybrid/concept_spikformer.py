# snn_research/models/hybrid/concept_spikformer.py
# ファイルパス: snn_research/models/hybrid/concept_spikformer.py
# 日本語タイトル: 概念統合型Spikformer v2 (With Projection Head)
# 修正: 画像と概念の埋め込みを直接比較するのではなく、Projection Headを通して
#       共通の潜在空間に写像してから比較するように変更。これにより「Balloonバイアス」を解消する。

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from snn_research.models.transformer.spikformer import Spikformer
from snn_research.core.neurons.da_lif_node import DualAdaptiveLIFNode
from spikingjelly.activation_based import layer, functional as SJ_F

class ConceptCrossAttention(nn.Module):
    """
    視覚情報(Query)と概念情報(Key/Value)のCross-Attentionを行うSNNモジュール。
    """
    def __init__(self, dim: int, num_heads: int = 4, tau_m: float = 2.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q_linear = layer.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)
        
        self.k_linear = layer.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)
        
        self.v_linear = layer.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)
        
        self.attn_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, v_threshold=0.5, detach_reset=True)
        
        self.proj = layer.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.last_attn_map: Optional[torch.Tensor] = None

    def forward(self, x_visual: torch.Tensor, x_concept: torch.Tensor) -> torch.Tensor:
        B, N, C = x_visual.shape
        
        if x_concept.dim() == 2:
            x_concept = x_concept.unsqueeze(1)
            
        M = x_concept.shape[1]

        q = self.q_lif(self.q_bn(self.q_linear(x_visual).transpose(1, 2)).transpose(1, 2))
        k = self.k_lif(self.k_bn(self.k_linear(x_concept).transpose(1, 2)).transpose(1, 2))
        v = self.v_lif(self.v_bn(self.v_linear(x_concept).transpose(1, 2)).transpose(1, 2))

        head_dim = C // self.num_heads
        q = q.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, M, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        self.last_attn_map = attn.detach()

        x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.attn_lif(x)
        
        x = self.proj_lif(self.proj_bn(self.proj(x).transpose(1, 2)).transpose(1, 2))
        return x

class ConceptSpikformer(nn.Module):
    """
    Spikformer + Concept Integration.
    v2: Projection Heads for CLIP-style embedding alignment.
    """
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 128,
        concept_dim: int = 64,
        num_classes: int = 10,
        num_layers: int = 2,
        num_heads: int = 4,
        projection_dim: int = 64 # [New] 共通埋め込み空間の次元
    ):
        super().__init__()
        
        # 1. Visual Stream
        self.visual_encoder = Spikformer(
            img_size_h=img_size, img_size_w=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=4,
            T=1,
            num_classes=0
        )
        
        # 2. Conceptual Stream (Input Processing)
        self.concept_input_proj = nn.Sequential(
            nn.Linear(concept_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            DualAdaptiveLIFNode(tau_m_init=2.0, detach_reset=True)
        )
        
        # 3. Integration (Attention)
        self.fusion_attn = ConceptCrossAttention(dim=embed_dim, num_heads=num_heads)
        self.norm_fusion = nn.LayerNorm(embed_dim)
        
        # 4. Classification Head (Specific Task)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 5. [New] Projection Heads for Contrastive Learning
        # これにより、画像と概念を「比較専用の空間」に射影する
        self.visual_proj = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.concept_proj = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.last_internal_state: Optional[torch.Tensor] = None
        self.last_projected_state: Optional[torch.Tensor] = None # For contrastive loss

    def forward_sensory(self, x: torch.Tensor) -> torch.Tensor:
        SJ_F.reset_net(self)
        if x.dim() == 5: x = x.squeeze(1)
        visual_feats = self.visual_encoder.forward_features(x)
        return visual_feats

    def forward_conceptual(self, concept_spikes: torch.Tensor) -> torch.Tensor:
        # 入力された概念スパイクを内部表現に変換
        concept_feat = self.concept_input_proj(concept_spikes)
        
        # [New] 対照学習用にさらに射影する
        # トレーナー側でこの戻り値と get_internal_state() を比較するが、
        # 次元を合わせるため、ここでは射影前の特徴を返し、
        # トレーナー側での比較時に射影後の値を使わせる設計にするのが一般的だが、
        # 今回の ConceptAugmentedTrainer のインターフェースに合わせて、
        # forward_conceptual は「射影後」の値を返すようにする。
        
        # ただし、Attention機構(integrate)には「射影前(embed_dim)」が必要なので、
        # ここは「射影前」を返し、Contrastive Loss計算時に「射影後」を使うようにメソッドを分けるか、
        # あるいは「射影前」を返す仕様を維持し、モデル内部で射影値を保持する。
        
        # 設計方針: forward_conceptual は Attention用の embed_dim (128) を返す。
        # 概念比較用のベクトル取得メソッドを別途用意する。
        return concept_feat

    def get_concept_projection(self, concept_feat: torch.Tensor) -> torch.Tensor:
        """概念特徴量(128)を共通空間(64)に射影して返す"""
        return self.concept_proj(concept_feat)

    def integrate(self, sensory_rep: torch.Tensor, conceptual_rep: Optional[torch.Tensor]) -> torch.Tensor:
        if conceptual_rep is not None:
            contextualized_feats = self.fusion_attn(sensory_rep, conceptual_rep)
            integrated = sensory_rep + contextualized_feats
        else:
            integrated = sensory_rep
            
        out_vec = integrated.mean(dim=1)
        out_vec = self.norm_fusion(out_vec)
        
        self.last_internal_state = out_vec
        
        # [New] 画像特徴を共通空間に射影して保存
        self.last_projected_state = self.visual_proj(out_vec)
        
        return self.head(out_vec)

    def get_internal_state(self) -> torch.Tensor:
        """
        トレーナーが呼び出すメソッド。
        Contrastive Lossの計算には「射影後」のベクトルを使うべき。
        """
        if self.last_projected_state is None:
            raise RuntimeError("Run forward/integrate first")
        return self.last_projected_state

    def get_last_attention_map(self) -> Optional[torch.Tensor]:
        return self.fusion_attn.last_attn_map
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sensory = self.forward_sensory(x)
        return self.integrate(sensory, None)