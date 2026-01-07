import torch
import torch.nn as nn
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork


def test_astrocyte_energy_management():
    astro = AstrocyteNetwork(initial_energy=100.0)
    assert astro.request_resource("visual_cortex", 10.0) is True
    assert astro.energy < 100.0

    # Simulate depletion
    astro.energy = 0.0
    assert astro.request_resource("visual_cortex", 10.0) is False


def test_homeostasis_scaling():
    astro = AstrocyteNetwork()
    # Force high glutamate to trigger downregulation
    astro.modulators["glutamate"] = 0.9

    model = nn.Linear(10, 10)
    original_weight = model.weight.clone()

    astro.maintain_homeostasis(model, learning_rate=0.1)

    # Should scale down (factor < 1.0)
    # 1.0 - 0.1 = 0.9
    assert torch.allclose(model.weight, original_weight * 0.9)


def test_neuron_death():
    astro = AstrocyteNetwork(initial_energy=100.0)
    layer = nn.Linear(10, 10)

    # Before death, no zeros (random init usually non-zero)
    assert (layer.weight == 0).sum() == 0

    # Simulate high death rate for visibility
    astro.handle_neuron_death(layer, death_rate=0.5)

    # Should have some zeros now
    assert (layer.weight == 0).sum() > 0
    # Energy should be consumed for repairs
    assert astro.energy < 100.0
