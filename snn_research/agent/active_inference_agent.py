# snn_research/agent/active_inference_agent.py
import torch
from typing import Dict, Any, cast, Union, Tuple
from snn_research.core.snn_core import SNNCore

class ActiveInferenceAgent:
    """
    Agent that minimizes Free Energy (Surprise) via Action.
    """
    def __init__(self, brain: SNNCore):
        self.snn_core = brain

    def step(self, observation: torch.Tensor) -> torch.Tensor:
        prediction = self.snn_core(observation)
        
        firing_rates = self.snn_core.get_firing_rates()
        surprise = sum(firing_rates.values()) if firing_rates else 0.0
        
        prediction_tensor: torch.Tensor
        if isinstance(prediction, tuple):
            prediction_tensor = prediction[0]
        else:
            prediction_tensor = prediction
            
        action = self._select_action_from_prediction(prediction_tensor, surprise)
        return action

    def _select_action_from_prediction(self, prediction: torch.Tensor, surprise: float) -> torch.Tensor:
        return torch.argmax(prediction, dim=1, keepdim=True)