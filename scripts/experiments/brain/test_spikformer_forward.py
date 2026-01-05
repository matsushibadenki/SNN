
from snn_research.models.transformer.spikformer import Spikformer
import torch
import sys
import os

# Create a minimal test without unittest framework first to debug easily

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)


def test_spikformer_forward():
    print("Testing Spikformer Forward Pass...")

    # Parameters
    B = 2
    T = 4
    C = 3
    H = 32
    W = 32
    num_classes = 10

    model = Spikformer(
        img_size_h=H,
        img_size_w=W,
        patch_size=4,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2,
        T=T,
        num_classes=num_classes
    )

    # Test 1: Static Input (B, C, H, W)
    x = torch.randn(B, C, H, W)
    y = model(x)
    print(f"Static Input {x.shape} -> Output {y.shape}")
    assert y.shape == (B, num_classes)

    # Test 2: Temporal Input (B, T, C, H, W)
    x_seq = torch.randn(B, T, C, H, W)
    y_seq = model(x_seq)
    print(f"Temporal Input {x_seq.shape} -> Output {y_seq.shape}")
    assert y_seq.shape == (B, num_classes)

    print("Spikformer Forward Pass Successful!")


if __name__ == "__main__":
    test_spikformer_forward()
