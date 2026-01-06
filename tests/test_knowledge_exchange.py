import torch
import torch.nn as nn
from snn_research.collective.knowledge_exchange import KnowledgeExchanger


def test_aggregate_weights():
    # Simple model for testing updates
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)

    exchanger = KnowledgeExchanger(agent_id="test_agent")

    # Simulate Peer weights
    # Peer 1: all 2.0
    peer1 = {"weight": torch.full((2, 2), 2.0), "bias": torch.full((2,), 2.0)}
    # Peer 2: all 3.0
    peer2 = {"weight": torch.full((2, 2), 3.0), "bias": torch.full((2,), 3.0)}

    # Peer Average should be (2.0 + 3.0) / 2 = 2.5

    # Apply aggregation with alpha = 0.5 (50% retention of old weights)
    # New weight = 0.5 * 1.0 + 0.5 * 2.5 = 0.5 + 1.25 = 1.75

    exchanger.aggregate_weights(model, [peer1, peer2], alpha=0.5)

    assert torch.allclose(model.weight, torch.full((2, 2), 1.75))
    assert torch.allclose(model.bias, torch.full((2,), 1.75))


def test_create_concept_packet():
    exchanger = KnowledgeExchanger(agent_id="agent_A")
    centroid = torch.tensor([0.1, 0.2, 0.3])
    packet = exchanger.create_concept_packet(
        "concept_123", centroid, "A test concept")

    assert packet["type"] == "concept_share"
    assert packet["sender"] == "agent_A"
    assert packet["concept_id"] == "concept_123"
    import pytest
    assert packet["centroid"] == pytest.approx([0.1, 0.2, 0.3])
