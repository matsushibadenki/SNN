import pytest
import torch
import torch.nn as nn
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator


class MockBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 2)

    def forward(self, x, **kwargs):
        # Return mock logits
        return torch.randn(1, 2)


def test_hebbian_reinforcement():
    brain = MockBrain()
    consolidator = SleepConsolidator(
        memory_system=None, target_brain_model=brain)

    # Capture initial weights
    initial_sum = brain.layer.weight.data.abs().sum().item()

    # Apply reinforcement with high clarity
    consolidator._apply_hebbian_reinforcement(strength=1.0)

    # Check if weights increased (absolute value)
    # w_new = w + 0.0001 * w = 1.0001 * w
    # sum(|w_new|) = 1.0001 * sum(|w|)
    new_sum = brain.layer.weight.data.abs().sum().item()

    assert new_sum > initial_sum
    assert new_sum == pytest.approx(initial_sum * 1.0001, rel=1e-5)
