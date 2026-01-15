# scripts/experiments/social/run_phase5_social_learning.py
import torch
import logging
from typing import Any, Tuple, Union

logger = logging.getLogger(__name__)

class SocialAgent:
    def __init__(self, brain: Any):
        self.brain = brain

    def listen(self, auditory_input: torch.Tensor) -> torch.Tensor:
        # SNNCore might return (output, spikes, mems)
        brain_output = self.brain(auditory_input)
        
        logits: torch.Tensor
        if isinstance(brain_output, tuple):
            logits = brain_output[0]
        else:
            logits = brain_output
            
        # Fix: Ensure logits is a Tensor before softmax
        response = torch.softmax(logits, dim=1)
        return response

    def learn(self, stimulus: torch.Tensor, reward: float):
        # Fix: self.brain is likely SNNCore which now has update_plasticity
        if hasattr(self.brain, "update_plasticity"):
            self.brain.update_plasticity(reward)