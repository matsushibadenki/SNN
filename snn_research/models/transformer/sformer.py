# ファイルパス: snn_research/architectures/sformer.py
# Title: SFormer (Scale-and-Fire Transformer) - High Fidelity T=1 Implementation
# Description:
#   ROADMAP Phase 3「究極の低遅延バックボーン」の完全実装。
#   Scale-and-Fire Neuron (SFN) をQuery, Key, Value, FFNの全段に適用し、
#   QK-Normによる安定化と合わせて T=1 での高精度動作を実現する。
#   Softmaxアテンションの前段階で入力をSFNにより量子化することで、
#   ハードウェアフレンドリーな計算グラフを構築する。

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, cast
import logging
import math

from snn_research.core.base import BaseModel
from snn_research.core.neurons import ScaleAndFireNeuron
from snn_research.core.layers.qk_norm import SpikingQKNorm

logger = logging.getLogger(__name__)

class SFNAttention(nn.Module):
    """
    Scale-and-Fire Attention Mechanism.
    Q, K, V を SFN で量子化(スパイク化)してからアテンションスコアを計算する。
    T=1 SNNのための主要コンポーネント。
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        sf_threshold: float = 4.0,
        sf_levels: int = 8
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # QK-Norm (Phase 3 Key Component)
        # ヘッドごとに正規化するため、head_dim を指定
        self.qk_norm_q = SpikingQKNorm(self.head_dim)
        self.qk_norm_k = SpikingQKNorm(self.head_dim)

        # Scale-and-Fire Neurons for Q, K, V
        # これにより、行列積 (Q@K.T, Attn@V) の入力が整数(量子化値)になる
        self.sfn_q = ScaleAndFireNeuron(d_model, num_levels=sf_levels, base_threshold=sf_threshold)
        self.sfn_k = ScaleAndFireNeuron(d_model, num_levels=sf_levels, base_threshold=sf_threshold)
        self.sfn_v = ScaleAndFireNeuron(d_model, num_levels=sf_levels, base_threshold=sf_threshold)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        H = self.nhead
        Dh = self.head_dim

        # 1. Linear Projections
        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2) # (B, H, L, Dh)
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)

        # 2. Apply QK-Norm (Stabilization)
        q = self.qk_norm_q(q)
        k = self.qk_norm_k(k)

        # 3. Apply SFN (Quantization / Spiking)
        # (B, H, L, Dh) -> SFN -> (B, H, L, Dh)
        # SFNは入力を量子化して返す(擬似的なスパイク/整数表現)
        q, _ = self.sfn_q(q)
        k, _ = self.sfn_k(k)
        v, _ = self.sfn_v(v)

        # 4. Scaled Dot-Product Attention
        # Q, K, V は量子化されているため、ハードウェア上では効率的に計算可能
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask: (B, 1, 1, L) or similar
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Attn @ V
        attn_output = torch.matmul(attn_probs, v)

        # 5. Output Projection
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        output = self.out_proj(attn_output)

        return output

class SFormerBlock(nn.Module):
    """
    SFormerの基本ブロック。
    SFNによる活性化と、SFNAttentionによる高効率アテンションを特徴とする。
    """
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int, 
        dropout: float = 0.1,
        sf_threshold: float = 4.0,
        sf_levels: int = 8
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Attention Block (SFN Integrated)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SFNAttention(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            sf_threshold=sf_threshold, 
            sf_levels=sf_levels
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. FFN Block
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        
        # SFN (Scale-and-Fire Neuron) for FFN Activation (T=1)
        self.sfn_ffn = ScaleAndFireNeuron(
            features=dim_feedforward, 
            num_levels=sf_levels, 
            base_threshold=sf_threshold
        )
        
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # --- Attention Sub-layer ---
        shortcut = x
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, mask=mask)
        x = shortcut + self.dropout1(attn_out)
        
        # --- FFN Sub-layer ---
        shortcut = x
        x_norm = self.norm2(x)
        
        x_ff = self.linear1(x_norm)
        
        # SFN Activation
        # SFN returns (quantized_output, quantized_output) tuple in forward
        x_ff, _ = self.sfn_ffn(x_ff) 
        
        x_ff = self.linear2(x_ff)
        x = shortcut + self.dropout2(x_ff)
        
        return x

class SFormer(BaseModel):
    """
    Scale-and-Fire Transformer (SFormer).
    T=1 での高精度推論を実現する次世代SNNバックボーン。
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        neuron_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        super().__init__()
        # SFormerは本質的に T=1 モデルである
        self.time_steps = 1 
        
        if neuron_config is None:
            neuron_config = {}
            
        sf_levels = int(neuron_config.get('num_levels', 8)) # SFNの量子化レベル数
        sf_threshold = float(neuron_config.get('base_threshold', 4.0))

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            SFormerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                sf_threshold=sf_threshold,
                sf_levels=sf_levels
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        logger.info(f"✅ SFormer initialized (T=1, Levels={sf_levels}). High-Fidelity Mode.")

    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_spikes: bool = False, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        
        # Embedding
        x = self.embedding(input_ids)
        
        # Positional Encoding handling
        if L <= self.pos_encoder.shape[1]:
             x = x + self.pos_encoder[:, :L, :]
        else:
             # Truncate pos_encoder if sequence is too long
             x = x + self.pos_encoder[:, :self.pos_encoder.shape[1], :]

        x = self.dropout(x)
        
        # Layers (T=1 なので時間ループなし)
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        logits = self.output_projection(x)
        
        # 統計情報 (SFNの総スパイク数を取得)
        total_spikes = self.get_total_spikes()
        avg_spikes = torch.tensor(total_spikes / (B * L), device=x.device) if return_spikes else torch.tensor(0.0, device=x.device)
        mem = torch.tensor(0.0, device=x.device) # SFormerは膜電位を外部に出さない
        
        return logits, avg_spikes, mem