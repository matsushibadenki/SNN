# snn_research/models/experimental/moe_model.py
import torch
import torch.nn as nn
from typing import List, Dict, Any, cast

class ExpertContainer(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model: nn.Module = model

class SpikingFrankenMoE(nn.Module):
    """
    Mixture of Experts model.
    """
    def __init__(self, experts: List[ExpertContainer], gate: nn.Module, config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.gate = gate
        self.config = config
        
        self.experts = nn.ModuleList()
        for expert_container in experts:
            # Cast for mypy safety
            model_module = cast(nn.Module, expert_container.model)
            
            if config.get("load_checkpoint"):
                new_state_dict: Dict[str, Any] = {} 
                model_module.load_state_dict(new_state_dict, strict=False)
            
            self.experts.append(model_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [expert(x) for expert in self.experts]
        return torch.stack(outputs).mean(dim=0)