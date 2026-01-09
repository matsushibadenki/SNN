# ファイルパス: tests/test_sleep_consolidation.py
# 修正: float精度の問題で失敗するテストを修正 (double型を使用)

import pytest
import torch
import torch.nn as nn
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator


class MockBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Use double precision for testing small updates
        self.layer = nn.Linear(10, 2).double()

    def forward(self, x, **kwargs):
        # Return mock logits
        return torch.randn(1, 2, dtype=torch.float64)


def test_hebbian_reinforcement():
    brain = MockBrain()
    consolidator = SleepConsolidator(
        memory_system=None, target_brain_model=brain)

    # Capture initial weights
    initial_sum = brain.layer.weight.data.abs().sum().item()

    # Apply reinforcement with high clarity
    # Implementation: param.data += (1e-5 * strength) * param.data * 0.01
    # Total multiplier = 1 + (1e-5 * 1.0 * 0.01) = 1 + 1e-7
    # Double precision allows detecting 1e-7 change.
    consolidator._apply_hebbian_reinforcement(strength=1.0)

    new_sum = brain.layer.weight.data.abs().sum().item()

    assert new_sum > initial_sum

    # Expected multiplier: 1 + 1e-7
    expected_multiplier = 1.0 + (1e-5 * 0.01)
    assert new_sum == pytest.approx(
        initial_sum * expected_multiplier, rel=1e-8)
