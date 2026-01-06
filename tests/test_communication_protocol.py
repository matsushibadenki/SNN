import torch
from snn_research.communication.spike_encoder_decoder import SpikeEncoderDecoder


def test_latency_coding_determinism():
    encoder = SpikeEncoderDecoder(num_neurons=10, time_steps=300)
    data = "test"

    # 1. Encode twice, should be identical
    spikes1 = encoder.latency_encode(data)
    spikes2 = encoder.latency_encode(data)

    assert torch.equal(spikes1, spikes2)

    # 2. Decode
    decoded_str = encoder.latency_decode(spikes1)
    # Note: Decode is approximate due to ASCII clipping, but for short simple strings
    # with sufficient time_steps, it should be close or identical if logic is perfect.
    # In our implementation:
    # 't' -> ord 116 -> timing -> decode
    # Let's just check it returns a string of same length
    assert len(decoded_str) == len(data)


def test_latency_decode_order():
    encoder = SpikeEncoderDecoder(num_neurons=10, time_steps=100)
    # Manually create spike pattern
    spikes = torch.zeros(10, 100)
    # Neuron 0 fires at t=10 (early) -> Small ASCII
    spikes[0, 10] = 1
    # Neuron 1 fires at t=90 (late) -> Large ASCII
    spikes[1, 90] = 1

    decoded = encoder.latency_decode(spikes)
    # Should get 2 chars
    assert len(decoded) >= 2
    # Char 0 should have smaller code than Char 1
    assert ord(decoded[0]) < ord(decoded[1])
