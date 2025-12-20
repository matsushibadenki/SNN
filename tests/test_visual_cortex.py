
import torch
import unittest
from snn_research.models.bio.visual_cortex import VisualCortex

class TestVisualCortex(unittest.TestCase):
    def test_visual_cortex_static_image(self):
        # Test with static image (B, C, H, W)
        batch_size = 2
        channels = 3
        height = 32
        width = 32
        time_steps = 4
        
        model = VisualCortex(
            input_channels=channels,
            height=height,
            width=width,
            time_steps=time_steps,
            d_model=64,
            d_state=32
        )
        
        x = torch.randn(batch_size, channels, height, width)
        
        states, errors, recons = model(x)
        
        self.assertEqual(states.shape, (batch_size, time_steps, 32)) # D_State
        self.assertEqual(errors.shape, (batch_size, time_steps, 64)) # D_Model
        self.assertEqual(recons.shape, (batch_size, time_steps, 64)) # D_Model
        
    def test_visual_cortex_video_stream(self):
        # Test with video stream (B, T, C, H, W)
        batch_size = 2
        channels = 1
        height = 16
        width = 16
        time_steps = 5
        
        model = VisualCortex(
            input_channels=channels,
            height=height,
            width=width,
            time_steps=time_steps,
            d_model=32,
            d_state=16
        )
        
        x = torch.randn(batch_size, time_steps, channels, height, width)
        
        states, errors, recons = model(x)
        
        self.assertEqual(states.shape, (batch_size, time_steps, 16))
        self.assertEqual(errors.shape, (batch_size, time_steps, 32))

    def test_reset(self):
        model = VisualCortex(1, 8, 8, time_steps=2)
        model.reset_spike_stats()
        # Ensure no crash

if __name__ == '__main__':
    unittest.main()
