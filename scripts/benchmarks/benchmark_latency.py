import time
import torch
import logging
from omegaconf import OmegaConf
from snn_research.core.snn_core import SNNCore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark(config_path: str):
    logger.info(f"🚀 Benchmarking config: {config_path}")

    # Load config
    conf = OmegaConf.load(config_path)

    # Initialize model
    vocab_size = 1000
    model = SNNCore(config=conf.model, vocab_size=vocab_size,
                    backend="spikingjelly")
    model.eval()

    # Dummy input
    batch_size = 1
    seq_len = conf.model.time_steps
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Warmup
    logger.info("Warmup...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids)

    # Measure latency
    logger.info("Measuring latency...")
    latencies = []
    with torch.no_grad():
        for _ in range(20):
            start_time = time.time()
            _ = model(input_ids)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # ms

    avg_latency = sum(latencies) / len(latencies)
    logger.info(f"⚡️ Average Inference Latency: {avg_latency:.2f} ms")

    with open("latency_result.txt", "w") as f:
        f.write(f"Average Inference Latency: {avg_latency:.2f} ms\n")

    return avg_latency


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(
        sys.argv) > 1 else "configs/models/large_scale.yaml"
    benchmark(config_path)
