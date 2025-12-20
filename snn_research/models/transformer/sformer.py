# ファイルパス: snn_research/models/transformer/sformer.py
# Title: SFormer (Scale-and-Fire Transformer) - High Fidelity T=1 Implementation (Fixed v3.2)
# Description:
#   ROADMAP Phase 3「究極の低遅延バックボーン」の完全実装。
#   修正: 
#   1. SFNAttentionにおいて、SFNの適用タイミングをHead分割前に変更し、次元不整合を解消。
#   2. generate メソッドを追加し、自己回帰的なトークン生成(Thinking Process)を可能に。
#   3. generate メソッドの引数 (pad_token_id, eos_token_id) に対応し、ReasoningEngineとの互換性を修正。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import logging

from snn_research.core.base import BaseModel
from snn_research.core.neurons import ScaleAndFireNeuron
from snn_research.core.layers.qk_norm import SpikingQKNorm

logger = logging.getLogger(__name__)

class SFNAttention(nn.Module):
    """
    Scale-and-Fire Attention Mechanism.
    Q, K, V を SFN で量子化(スパイク化)してからアテンションスコアを計算する。
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

        # QK-Norm (Head次元に対して適用)
        self.qk_norm_q = SpikingQKNorm(self.head_dim)
        self.qk_norm_k = SpikingQKNorm(self.head_dim)

        # Scale-and-Fire Neurons (d_model全体に対して適用するため、Head分割前に配置)
        self.sfn_q = ScaleAndFireNeuron(d_model, num_levels=sf_levels, base_threshold=sf_threshold)
        self.sfn_k = ScaleAndFireNeuron(d_model, num_levels=sf_levels, base_threshold=sf_threshold)
        self.sfn_v = ScaleAndFireNeuron(d_model, num_levels=sf_levels, base_threshold=sf_threshold)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        H = self.nhead
        Dh = self.head_dim

        # 1. Linear Projections: (B, L, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Apply SFN (Quantization): (B, L, D)
        # SFNは (B, L, C) の形状を受け付け、C=d_model と一致することを確認する
        q, _ = self.sfn_q(q)
        k, _ = self.sfn_k(k)
        v, _ = self.sfn_v(v)

        # 3. Head Split: (B, L, D) -> (B, L, H, Dh) -> (B, H, L, Dh)
        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        # 4. Apply QK-Norm: (B, H, L, Dh)
        # QK-Normは最後の次元(Dh)に対して正規化を行う
        q = self.qk_norm_q(q)
        k = self.qk_norm_k(k)

        # 5. Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # --- マスク適用 ---
        if mask is not None:
            # mask形状: (L, L) または (B, 1, L, L) を想定
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Attn @ V
        attn_output = torch.matmul(attn_probs, v)

        # 6. Output Projection
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        output = self.out_proj(attn_output)

        return output

class SFormerBlock(nn.Module):
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
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SFNAttention(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            sf_threshold=sf_threshold, 
            sf_levels=sf_levels
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        
        self.sfn_ffn = ScaleAndFireNeuron(
            features=dim_feedforward, 
            num_levels=sf_levels, 
            base_threshold=sf_threshold
        )
        
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shortcut = x
        x_norm = self.norm1(x)
        # マスクをAttentionに渡す
        attn_out = self.attn(x_norm, mask=mask)
        x = shortcut + self.dropout1(attn_out)
        
        shortcut = x
        x_norm = self.norm2(x)
        
        x_ff = self.linear1(x_norm)
        x_ff, _ = self.sfn_ffn(x_ff) 
        x_ff = self.linear2(x_ff)
        x = shortcut + self.dropout2(x_ff)
        
        return x

class SFormer(BaseModel):
    """
    Scale-and-Fire Transformer (SFormer).
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
        self.time_steps = 1 
        
        if neuron_config is None:
            neuron_config = {}
            
        sf_levels = int(neuron_config.get('num_levels', 8))
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
        device = input_ids.device
        
        # Embedding
        x = self.embedding(input_ids)
        
        # Positional Encoding Handling
        max_len = self.pos_encoder.shape[1]
        if L <= max_len:
             x = x + self.pos_encoder[:, :L, :]
        else:
             # シーケンスが長すぎる場合、PosEncは適用可能な範囲(max_len)まで適用し、
             # それ以降はPosEncなし（あるいは相対位置などが望ましいがここでは簡易実装）とする
             pos_enc = self.pos_encoder[:, :max_len, :]
             x[:, :max_len, :] = x[:, :max_len, :] + pos_enc

        x = self.dropout(x)
        
        # --- 因果マスクの生成 ---
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1) == 0
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Layers
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
            
        x = self.norm(x)
        logits = self.output_projection(x)
        
        total_spikes = self.get_total_spikes()
        avg_spikes = torch.tensor(total_spikes / (B * L + 1e-8), device=x.device) if return_spikes else torch.tensor(0.0, device=x.device)
        mem = torch.tensor(0.0, device=x.device)
        
        return logits, avg_spikes, mem

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_length: int, 
        temperature: float = 1.0, 
        do_sample: bool = True,
        top_k: int = 50,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        自己回帰的なテキスト生成ループ。
        
        Args:
            input_ids: (Batch, SeqLen) の開始トークン列
            max_length: 生成後の最大シーケンス長
            temperature: サンプリング温度
            do_sample: Trueならサンプリング、FalseならGreedy
            top_k: Top-KサンプリングのK
            pad_token_id: パディングトークンID (無視されるが互換性のために維持)
            eos_token_id: 生成終了トークンID
            **kwargs: その他の引数 (互換性のため)
            
        Returns:
            generated_ids: (Batch, MaxLength)
        """
        self.eval()
        curr_ids = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            # Context Windowの制限 (PosEncの長さに合わせる)
            cond_ids = curr_ids[:, -self.pos_encoder.size(1):]
            
            # Forward pass
            logits, _, _ = self.forward(cond_ids)
            next_token_logits = logits[:, -1, :] # 最後のトークンの予測
            
            # Temperature
            next_token_logits = next_token_logits / max(temperature, 1e-5)
            
            if do_sample:
                # Top-K Filtering
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # EOS Check
            if eos_token_id is not None:
                # バッチサイズ1を想定しているが、安全のため全要素チェック
                if (next_token == eos_token_id).all():
                    break
                    
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            if curr_ids.size(1) >= max_length:
                break
                
        return curr_ids