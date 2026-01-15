# scripts/benchmarks/run_benchmark_suite.py
import logging
import torch
import time
import psutil
import os
import sys
from typing import Dict, Any, cast # Added cast

# Suppress Logs Early
for lib in ['spikingjelly', 'spikingjelly.activation_based.base']:
    logging.getLogger(lib).setLevel(logging.ERROR)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from snn_research.models.transformer.spiking_transformer import SpikingTransformerV2
from snn_research.core.snn_core import SNNCore

# --- Phase 2 Objectives ---
TARGET_LATENCY_MS = 10.0  # Goal: < 10ms (Ideally < 5ms)

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

class BenchmarkSuite:
    def __init__(self):
        self.device = self._detect_device()
        self.results: Dict[str, Any] = {}
        print(f"🚀 Benchmark Suite Initialized (Device: {self.device})")

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def measure_throughput_latency(self, model: torch.nn.Module, input_shape: tuple, batch_size: int = 1) -> tuple[float, float]:
        model.eval()
        model.to(self.device)
        
        # Prepare Dummy Input
        is_transformer = hasattr(model, 'input_ids') or 'Transformer' in model.__class__.__name__
        if is_transformer:
            dummy_input = torch.randint(0, 100, (batch_size, input_shape[0])).to(self.device)
        else:
            dummy_input = torch.randn(batch_size, *input_shape).to(self.device)

        # check reset_state
        if hasattr(model, 'reset_state'):
            # Cast model to Any to avoid mypy error "Tensor not callable"
            cast(Any, model).reset_state()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                if is_transformer: _ = model(input_ids=dummy_input)
                else: _ = model(dummy_input)

        if self.device == "cuda": torch.cuda.synchronize()
        elif self.device == "mps": torch.mps.synchronize()

        # Measurement
        iterations = 50
        start_time = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                if is_transformer: _ = model(input_ids=dummy_input)
                else: _ = model(dummy_input)
        
        if self.device == "cuda": torch.cuda.synchronize()
        elif self.device == "mps": torch.mps.synchronize()

        end_time = time.time()
        total_time = end_time - start_time
        
        avg_latency_ms = (total_time / iterations) * 1000
        avg_latency_per_sample = avg_latency_ms / batch_size
        throughput = (batch_size * iterations) / total_time

        return throughput, avg_latency_per_sample

    def run_core_benchmarks(self):
        print("\n--- Running Core SNN Benchmarks ---")
        try:
            config = {
                "architecture_type": "hybrid",
                "hidden_features": 128,
                "in_features": 64,
                "out_features": 10,
                "tau": 2.0, "threshold": 1.0
            }
            model = SNNCore(config=config, vocab_size=10)
            tput, latency = self.measure_throughput_latency(model, (64,), batch_size=1)

            self.results["SNN_Core"] = f"{latency:.4f} ms"
            status = "✅ PASS" if latency < TARGET_LATENCY_MS else "⚠️ WARN"
            print(f"{status} SNN Core: {latency:.4f} ms (Target < {TARGET_LATENCY_MS} ms)")

        except Exception as e:
            print(f"❌ Error in Core SNN Benchmark: {e}")

    def run_transformer_benchmarks(self):
        print("\n--- Running SFormer (T=1) Benchmarks ---")
        try:
            neuron_config = {'tau_mem': 2.0, 'base_threshold': 1.0}
            model = SpikingTransformerV2(
                vocab_size=100, d_model=64, nhead=4, num_encoder_layers=2,
                dim_feedforward=128, time_steps=1, neuron_config=neuron_config,
                img_size=64, patch_size=16
            )
            tput, latency = self.measure_throughput_latency(model, (64,), batch_size=1)
            
            self.results["SFormer_T1"] = f"{latency:.4f} ms"
            status = "✅ PASS" if latency < TARGET_LATENCY_MS else "⚠️ WARN"
            print(f"{status} SFormer (T=1): {latency:.4f} ms (Target < {TARGET_LATENCY_MS} ms)")
        except Exception as e:
            print(f"❌ Error in SFormer Benchmark: {e}")

    def save_report(self):
        print("\n===========================================")
        print("   🏆 Phase 2 Benchmark Report Summary     ")
        print("===========================================")
        for k, v in self.results.items():
            print(f"🔹 {k}: {v}")
        print("===========================================\n")

if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_core_benchmarks()
    suite.run_transformer_benchmarks()
    suite.save_report()