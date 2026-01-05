# ファイルパス: snn_research/models/experimental/semm_model.py
# Title: Spiking Experts Mixture Mechanism (SEMM) with T=1 SFN Router
# Description:
#   Phase 3 実装の強化版。
#   修正: forward メソッドで因果マスクを生成し、エキスパートに渡すように変更。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from snn_research.core.base import BaseModel
from snn_research.core.neurons import ScaleAndFireNeuron
from snn_research.models.transformer.sformer import SFormerBlock
import logging

logger = logging.getLogger(__name__)

class SpikingRouter(nn.Module):
    """
    SFN (Scale-and-Fire Neuron) を使用したスパイクベースルーター。
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, sf_threshold: float = 4.0):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        
        self.sfn = ScaleAndFireNeuron(
            features=num_experts, 
            num_levels=4, 
            base_threshold=sf_threshold
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        topk_values, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_values)
        routing_input = F.relu(mask) 
        routing_weights, _ = self.sfn(routing_input)
        return routing_weights, logits

class SEMMBlock(nn.Module):
    """
    1つのSEMM層。SFormerBlockをエキスパートとして持つ。
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int, dim_feedforward: int):
        super().__init__()
        self.router = SpikingRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            SFormerBlock(
                d_model=d_model, 
                nhead=4, 
                dim_feedforward=dim_feedforward
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        routing_weights, router_logits = self.router(x)
        final_output = torch.zeros_like(x)
        
        for i, expert in enumerate(self.experts):
            gate = routing_weights[:, :, i].unsqueeze(-1)
            
            if gate.sum() > 0:
                # マスクをエキスパートに渡す
                expert_out = expert(x, mask=mask)
                final_output += expert_out * gate
            
        return final_output, router_logits

class SEMMModel(BaseModel):
    """
    Spiking Experts Mixture Mechanism (SEMM) Model.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_experts: int = 4,
        top_k: int = 2,
        neuron_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        super().__init__()
        self.time_steps = 1
        self.num_experts = num_experts
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        dim_feedforward = d_model * 4
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SEMMBlock(
                d_model=d_model, 
                num_experts=num_experts, 
                top_k=top_k,
                dim_feedforward=dim_feedforward
            ))
            
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_weights()
        logger.info("✅ SEMM Model (T=1, SFN Router) initialized.")

    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_spikes: bool = False, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x = self.embedding(input_ids)
        B, L, D = x.shape
        
        # --- 因果マスク生成 ---
        device = x.device
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1) == 0
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        all_router_logits = []
        
        for layer in self.layers:
            # マスクを渡す
            x, router_logits = layer(x, mask=causal_mask)
            all_router_logits.append(router_logits)
            
        logits = self.output_projection(x)
        aux_loss_logits = torch.stack(all_router_logits, dim=2)
        
        avg_spikes = torch.tensor(0.0, device=x.device)
        if return_spikes:
            avg_spikes = torch.tensor(self.get_total_spikes() / input_ids.numel(), device=x.device)
            
        return logits, avg_spikes, torch.tensor(0.0, device=x.device), aux_loss_logits