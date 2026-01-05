
import torch
import unittest
from snn_research.core.layers.qk_norm import SpikingQKNorm


class TestSpikingQKNorm(unittest.TestCase):
    def test_qk_norm_2d(self):
        batch_size = 4
        dim = 16
        layer = SpikingQKNorm(dim=dim)

        x = torch.randn(batch_size, dim)
        out = layer(x)

        # Check output shape
        self.assertEqual(out.shape, (batch_size, dim))

        # Manual calculation
        mean_sq = (x ** 2).mean(dim=-1, keepdim=True)
        rsqrt = 1.0 / torch.sqrt(mean_sq + layer.eps)
        expected = x * rsqrt * layer.scale

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_qk_norm_3d(self):
        batch_size = 2
        time_steps = 5
        dim = 8
        layer = SpikingQKNorm(dim=dim)

        x = torch.randn(batch_size, time_steps, dim)
        out = layer(x)

        # Check output shape
        self.assertEqual(out.shape, (batch_size, time_steps, dim))

        # Manual calculation
        mean_sq = (x ** 2).mean(dim=-1, keepdim=True)
        rsqrt = 1.0 / torch.sqrt(mean_sq + layer.eps)
        expected = x * rsqrt * layer.scale

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_gradients(self):
        dim = 8
        layer = SpikingQKNorm(dim=dim)
        x = torch.randn(2, dim, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.scale.grad)

    def test_spikes_input(self):
        # Test with binary input (0, 1) to simulate spikes
        dim = 4
        layer = SpikingQKNorm(dim=dim)
        x = torch.randint(0, 2, (2, dim)).float()
        layer(x)

        # If x is all zeros, output should be 0 (due to eps handling in rsqrt)
        # But rsqrt(0 + eps) = 1/sqrt(eps). 0 * large = 0.
        x_zeros = torch.zeros(1, dim)
        out_zeros = layer(x_zeros)
        self.assertTrue(torch.allclose(out_zeros, torch.zeros_like(out_zeros)))


if __name__ == '__main__':
    unittest.main()
