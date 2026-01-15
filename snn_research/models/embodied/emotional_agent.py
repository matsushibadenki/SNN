# snn_research/models/embodied/emotional_agent.py
# 修正: self.brain の型キャストを追加し、Mypyエラーを解消。

import torch
import torch.nn as nn
from typing import Tuple, Optional, Any, cast

class MotorCortex(nn.Module):
    def __init__(self, input_dim: int = 129, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, cortex_state: torch.Tensor, emotion_value: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([cortex_state, emotion_value], dim=1)
        logits = self.network(combined)
        return logits

class EmotionalAgent(nn.Module):
    def __init__(self, brain: nn.Module):
        super().__init__()
        self.brain = brain
        self.motor = MotorCortex(input_dim=128+1, hidden_dim=64, output_dim=2)
        
    def act(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 認知 & 感情
        _ = self.brain(img)
        
        # Mypy対策: brainを動的型として扱う
        brain_dynamic = cast(Any, self.brain)
        
        if hasattr(brain_dynamic, "get_internal_state"):
            internal_state = brain_dynamic.get_internal_state()
        else:
            raise RuntimeError("Brain model must have 'get_internal_state' method.")

        _, emotion = self.brain(img)
        
        # 2. 行動選択
        action_logits = self.motor(internal_state, emotion)
        
        return action_logits, emotion