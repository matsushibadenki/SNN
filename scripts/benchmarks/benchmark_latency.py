# ファイルパス: scripts/benchmarks/benchmark_latency.py
# Title: Latency Benchmark
# 修正内容: Mypyエラー修正 (Argument type mismatch)。

import torch
import time
import logging
import sys
import os
from typing import Dict, Any, cast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from snn_research.core.snn_core import SNNCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark():
    logger.info("⏱️ Starting Latency Benchmark...")
    
    device = "cpu"
    vocab_size = 100
    
    # Mock config object (DictConfig usually)
    model_conf = {
        "hidden_dim": 128,
        "layers": 2
    }
    
    try:
        # [Mypy Fix] config引数に対して明示的にDict[str, Any]へキャスト
        model = SNNCore(
            config=cast(Dict[str, Any], model_conf), 
            vocab_size=vocab_size,
            hidden_features=128,
            out_features=10
        ).to(device)
        
        input_tensor = torch.randn(1, 64).to(device)
        
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)
            
        # Measurement
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # ms
            
        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"Average Latency: {avg_latency:.4f} ms")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")

if __name__ == "__main__":
    benchmark()