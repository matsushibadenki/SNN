import torch
from snn_research.core.neurons.lif_neuron import LIFNeuron


def test_lif_neuron_dynamics():
    features = 10
    neuron = LIFNeuron(features=features, tau_mem=10.0,
                       v_threshold=1.0, dt=1.0)

    # Check initialization
    assert neuron.membrane_potential is None

    # Input that should NOT trigger spike
    input_current = torch.full((1, features), 0.05)  # Small input

    spike, mem = neuron(input_current)

    assert spike.sum() == 0
    assert torch.all(mem > 0)  # Membrane potential should increase
    assert torch.all(mem < 1.0)  # But not cross threshold

    # Input that SHOULD trigger spike (large input)
    input_current_large = torch.full((1, features), 2.0)

    # Step multiple times
    for _ in range(5):
        spike, mem = neuron(input_current_large)

    # Eventually should spike
    assert spike.sum() > 0
    # And verify reset happens (some potentials should be 0.0 or close to reset)
    # Note: Logic is Hard Reset to v_reset=0.0
    assert torch.any(mem == 0.0)


def test_lif_neuron_stateful():
    neuron = LIFNeuron(features=1)
    neuron.set_stateful(True)

    neuron(torch.ones(1, 1))
    assert neuron.membrane_potential is not None

    # Reset should clear state
    neuron.reset()
    assert neuron.membrane_potential is None
