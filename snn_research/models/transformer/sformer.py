# ファイルパス: snn_research/models/transformer/sformer.py
# Title: SFormer (Scale-and-Fire Transformer) - High Fidelity T=1 Implementation (Fixed v3.5)
# Description:
#   ROADMAP Phase 3「究極の低遅延バックボーン」の完全実装。
#   修正 v3.5: MPS環境での "Placeholder storage error" を回避するため、
#             forward入力時と主要な変形操作前に .contiguous() を徹底。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import logging

from snn_research.core.base import BaseModel
from snn_research.core.neurons import ScaleAndFireNeuron
from snn_research.core.layers.qk_norm import SpikingQKNorm

logger = logging.getLogger(__name__)

class SFNAttention(nn.Module):
    """
    Scale-and-Fire Attention Mechanism.
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

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.qk_norm_q = SpikingQKNorm(self.head_dim)
        self.qk_norm_k = SpikingQKNorm(self.head_dim)

        self.sfn_q = ScaleAndFireNeuron(d_model, num_levels=sf_levels, base_threshold=sf_threshold)
        self.sfn_k = ScaleAndFireNeuron(d_model, num_levels=sf_levels, base_threshold=sf_threshold)
        self.sfn_v = ScaleAndFireNeuron(d_model, num_levels=sf_levels, base_threshold=sf_threshold)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # MPS対策: 入力を整列
        x = x.contiguous()
        B, L, D = x.shape
        H = self.nhead
        Dh = self.head_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_out = self.sfn_q(q)
        q = q_out[0] if isinstance(q_out, tuple) else q_out
        
        k_out = self.sfn_k(k)
        k = k_out[0] if isinstance(k_out, tuple) else k_out
        
        v_out = self.sfn_v(v)
        v = v_out[0] if isinstance(v_out, tuple) else v_out

        # [MPS Critical Fix] transpose後は必ずcontiguous()
        q = q.view(B, L, H, Dh).transpose(1, 2).contiguous()
        k = k.view(B, L, H, Dh).transpose(1, 2).contiguous()
        v = v.view(B, L, H, Dh).transpose(1, 2).contiguous()

        q = self.qk_norm_q(q)
        k = self.qk_norm_k(k)

        # 安全のため k_t を作成して整列
        k_t = k.transpose(-2, -1).contiguous()
        attn_scores = torch.matmul(q, k_t) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)

        # Output Projection前も整列
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, L, D)
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
        attn_out = self.attn(x_norm, mask=mask)
        x = shortcut + self.dropout1(attn_out)
        
        shortcut = x
        x_norm = self.norm2(x)
        
        x_ff = self.linear1(x_norm)
        
        x_ff_out = self.sfn_ffn(x_ff)
        x_ff = x_ff_out[0] if isinstance(x_ff_out, tuple) else x_ff_out
        
        x_ff = self.linear2(x_ff)
        x = shortcut + self.dropout2(x_ff)
        
        return x

class SFormer(BaseModel):
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
        # [Critical Fix for MPS] Ensure input indices are contiguous before embedding lookup
        if not input_ids.is_contiguous():
            input_ids = input_ids.contiguous()
            
        B, L = input_ids.shape
        device = input_ids.device
        
        x = self.embedding(input_ids)
        
        max_len = self.pos_encoder.shape[1]
        if L <= max_len:
             x = x + self.pos_encoder[:, :L, :]
        else:
             pos_enc = self.pos_encoder[:, :max_len, :]
             x[:, :max_len, :] = x[:, :max_len, :] + pos_enc

        x = self.dropout(x)
        
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1) == 0
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask=causal_mask)
            
        x = self.norm(x)
        logits = self.output_projection(x)
        
        avg_spikes = torch.tensor(0.0, device=x.device)
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
        self.eval()
        curr_ids = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            cond_ids = curr_ids[:, -self.pos_encoder.size(1):]
            logits, _, _ = self.forward(cond_ids)
            next_token_logits = logits[:, -1, :]
            
            next_token_logits = next_token_logits / max(temperature, 1e-5)
            
            if do_sample:
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
                    
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            if curr_ids.size(1) >= max_length:
                break
                
        return curr_ids