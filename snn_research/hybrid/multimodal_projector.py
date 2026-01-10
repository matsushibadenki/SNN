# ファイルパス: snn_research/hybrid/multimodal_projector.py
# 日本語タイトル: Spiking Multimodal Projector (Vision-Language Bridge) v1.2
# 目的・内容:
#   ROADMAP Phase 2 "Multi-modal Integration" 対応。
#   既存のUnifiedSensoryProjector呼び出し（modality_configs, language_dim等）に対応し、
#   任意の数のモダリティを動的に扱えるように拡張。

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any, Union
import numpy as np

logger = logging.getLogger(__name__)

# BitNet対応: インポートできなければ通常のLinearを使用
try:
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    BitSpikeLinear = nn.Linear  # type: ignore

# DSA (Dynamic Sparse Attention) の利用
try:
    from snn_research.core.layers.dsa import DSALayer
except ImportError:
    DSALayer = None  # type: ignore


class CrossModalAttentionBlock(nn.Module):
    """
    異なるモダリティ間での情報のやり取りを行うブロック。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bitnet: bool = True
    ):
        super().__init__()
        self.d_model = d_model

        linear_cls = BitSpikeLinear if use_bitnet else nn.Linear

        self.q_proj = linear_cls(d_model, d_model)
        self.k_proj = linear_cls(d_model, d_model)
        self.v_proj = linear_cls(d_model, d_model)
        self.out_proj = linear_cls(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        _, T_ctx, _ = context.shape

        residual = x
        x = self.norm(x)

        q = self.q_proj(x).view(B, T, self.num_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, T_ctx, self.num_heads,
                                      self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, T_ctx, self.num_heads,
                                      self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return residual + self.out_proj(out)


class MultimodalProjector(nn.Module):
    """
    複数の感覚入力（Vision, Audio, Tactile等）と言語表現を統合するためのプロジェクター。

    Arguments:
        vision_dim (int, optional): レガシー引数 (Vision専用)
        text_dim (int, optional): レガシー引数 (Text専用)
        language_dim (int, optional): Text/Language共通埋め込み次元 (Brain v4互換)
        modality_configs (Dict[str, int], optional): 各モダリティの名前と入力次元のマップ
        embed_dim (int): 共通潜在空間の次元数
    """

    def __init__(
        self,
        vision_dim: Optional[int] = None,
        text_dim: Optional[int] = None,
        embed_dim: int = 512,
        use_bitnet: bool = True,
        # Legacy / Universal compatibility args
        language_dim: Optional[int] = None,
        modality_configs: Optional[Dict[str, int]] = None
    ):
        super().__init__()

        # 共通次元の決定 (embed_dim優先, 次にlanguage_dim)
        self.embed_dim = embed_dim if language_dim is None else language_dim
        linear_cls = BitSpikeLinear if use_bitnet else nn.Linear

        self.projections = nn.ModuleDict()

        # 1. レガシー設定 (Vision / Text)
        if vision_dim is not None:
            self.projections['vision'] = self._build_proj(
                vision_dim, self.embed_dim, linear_cls)

        if text_dim is not None:
            self.projections['text'] = self._build_proj(
                text_dim, self.embed_dim, linear_cls)

        # 2. 汎用設定 (Modality Configs)
        # Brain v4: {'vision': 784, 'tactile': 64, ...}
        if modality_configs is not None:
            for mod_name, input_dim in modality_configs.items():
                # 既に登録済みの場合はスキップ、または上書き
                self.projections[mod_name] = self._build_proj(
                    input_dim, self.embed_dim, linear_cls)

        # 3. Fusion Layers (Cross Attention)
        # 全てのモダリティ情報を統合するためのAttention
        # ここではシンプルに Self-Attention または Cross-Attention を利用
        self.fusion_attn = CrossModalAttentionBlock(
            d_model=self.embed_dim,
            num_heads=8,
            use_bitnet=use_bitnet
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        logger.info(
            f"🌉 Multimodal Projector initialized. Embed: {self.embed_dim}, Modalities: {list(self.projections.keys())}")

    def _build_proj(self, in_dim: int, out_dim: int, linear_cls: Any) -> nn.Sequential:
        return nn.Sequential(
            linear_cls(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            linear_cls(out_dim, out_dim)
        )

    def forward(self,
                inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
                text_features: Optional[torch.Tensor] = None) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Flexible forward method.

        Mode A: forward(vision_features, text_features) -> Compatible with old API
        Mode B: forward({'vision': ..., 'tactile': ...}) -> Compatible with Brain v4 (returns fused context)
        """

        # Mode A: Legacy (Vision, Text)
        if isinstance(inputs, torch.Tensor) and text_features is not None:
            vision_in = inputs
            v_emb = self.projections['vision'](vision_in)
            t_emb = self.projections['text'](text_features)

            # Simple Fusion
            fused = self.fusion_attn(v_emb, t_emb)

            # Pooling for Loss
            v_pool = v_emb.mean(dim=1)
            t_pool = t_emb.mean(dim=1)
            v_pool = v_pool / v_pool.norm(dim=-1, keepdim=True)
            t_pool = t_pool / t_pool.norm(dim=-1, keepdim=True)

            return {
                "vision_projected": v_emb,
                "text_projected": t_emb,
                "vision_pooled": v_pool,
                "text_pooled": t_pool,
                "fused_representation": fused
            }

        # Mode B: Universal (Dict input) -> Returns Fused Context Tensor [B, T_total, D]
        elif isinstance(inputs, dict):
            projected_features = []

            for mod_name, tensor in inputs.items():
                if mod_name in self.projections:
                    # [B, T, In_Dim] -> [B, T, Embed_Dim]
                    proj = self.projections[mod_name](tensor)
                    projected_features.append(proj)
                else:
                    logger.warning(
                        f"Unknown modality '{mod_name}' passed to Projector. Skipping.")

            if not projected_features:
                # Fallback for empty input (prevent crash)
                device = list(inputs.values())[
                    0].device if inputs else torch.device('cpu')
                return torch.zeros(1, 1, self.embed_dim, device=device)

            # Concatenate along time dimension [B, T_v + T_a + ..., D]
            concat_features = torch.cat(projected_features, dim=1)

            # Apply Self-Attention over all modalities (Fusion)
            # context is same as input for self-attention
            fused_context = self.fusion_attn(concat_features, concat_features)

            return fused_context

        else:
            raise ValueError("Invalid input format for MultimodalProjector")

    def compute_alignment_loss(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        v_feat = output["vision_pooled"]
        t_feat = output["text_pooled"]

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * v_feat @ t_feat.t()
        logits_per_text = logits_per_image.t()

        batch_size = v_feat.shape[0]
        labels = torch.arange(batch_size, device=v_feat.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        return (loss_i + loss_t) / 2


# Backward compatibility alias
UnifiedSensoryProjector = MultimodalProjector
