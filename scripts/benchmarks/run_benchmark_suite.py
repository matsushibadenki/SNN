# ファイルパス: scripts/benchmarks/run_benchmark_suite.py
# 日本語タイトル: SNN Benchmark Suite v2.4 (Mypy Fix)
# 目的: 動的属性アクセスにcastを使用して型エラーを回避。

import sys
import os
import torch
import torch.nn as nn
import logging
import time
import datetime
from typing import Dict, Any, cast, Optional
from omegaconf import OmegaConf

# プロジェクトルートの設定
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("Benchmark")

try:
    from snn_research.core.snn_core import SNNCore
except ImportError:
    SNNCore = None  # type: ignore

try:
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
except ImportError:
    BitSpikeMamba = None  # type: ignore


class BenchmarkSuite:
    def __init__(self, output_dir: str = "benchmarks/results"):
        print("⚙️ Initializing Benchmark Suite v2.4...")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"   -> Device selected: {self.device}")

        self.results: Dict[str, Any] = {
            "timestamp": str(datetime.datetime.now()),
            "hardware": self.device,
            "tests": {}
        }

    def _get_dummy_config(self, architecture: str) -> Dict[str, Any]:
        base_config = {
            "vocab_size": 100,
            "d_model": 64,
            "num_layers": 2,
            "time_steps": 16,
            "neuron_config": {"type": "lif", "base_threshold": 1.0}
        }
        if architecture == "bit_spike_mamba":
            base_config.update({
                "architecture_type": "bit_spike_mamba",
                "d_state": 16,
                "d_conv": 4,
                "expand": 2
            })
        else:
            base_config.update({
                "architecture_type": architecture,
                "nhead": 4,
                "dim_feedforward": 256
            })
        return base_config

    def _build_model(self, model_name: str, config_path: Optional[str] = None) -> nn.Module:
        if config_path and os.path.exists(config_path):
            conf = OmegaConf.load(config_path)
            model_config = OmegaConf.to_container(
                conf.model if 'model' in conf else conf, resolve=True)
        else:
            if "Mamba" in model_name:
                arch = "bit_spike_mamba"
            elif "DSA" in model_name:
                arch = "dsa_transformer"
            else:
                arch = "sformer"
            model_config = self._get_dummy_config(arch)

        model_config = cast(Dict[str, Any], model_config)
        vocab_size = int(model_config.get("vocab_size", 100))

        if model_config.get("architecture_type") == "bit_spike_mamba" and BitSpikeMamba is not None:
            return BitSpikeMamba(
                vocab_size=vocab_size,
                d_model=model_config["d_model"],
                d_state=model_config["d_state"],
                d_conv=model_config["d_conv"],
                expand=model_config["expand"],
                num_layers=model_config["num_layers"],
                time_steps=model_config.get("time_steps", 16),
                neuron_config=model_config["neuron_config"]
            ).to(self.device)

        if SNNCore is not None:
            return SNNCore(config=model_config, vocab_size=vocab_size).to(self.device)

        raise ImportError("No suitable model class found.")

    def _measure_model_size(self, model: nn.Module) -> float:
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024**2

    def run_smoke_test(self, model_name: str, config_path: Optional[str] = None):
        print(f"\n🧪 [Smoke Test] {model_name} ... ", end="", flush=True)

        try:
            model = self._build_model(model_name, config_path)
            model.eval()
            size_mb = self._measure_model_size(model)

            # Cast to Any to access dynamic attributes safely
            safe_model = cast(Any, model)

            vocab_size = 100
            if hasattr(safe_model, 'vocab_size'):
                vocab_size = safe_model.vocab_size
            elif hasattr(safe_model, 'config') and isinstance(safe_model.config, dict):
                vocab_size = safe_model.config.get('vocab_size', 100)

            input_ids = torch.randint(0, vocab_size, (1, 16)).to(self.device)

            with torch.no_grad():
                _ = model(input_ids)

            print(f"✅ PASSED (Size: {size_mb:.2f} MB)")
            self.results["tests"][f"smoke_{model_name}"] = {
                "status": "PASSED",
                "model_size_mb": size_mb
            }

        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.results["tests"][f"smoke_{model_name}"] = {
                "status": "FAILED", "error": str(e)}

    def run_efficiency_benchmark(self, model_name: str):
        print(
            f"\n⚡ [Efficiency Test] {model_name} (T=1 Latency) ... ", end="", flush=True)
        try:
            model = self._build_model(model_name)
            safe_model = cast(Any, model)

            if hasattr(safe_model, 'time_steps'):
                safe_model.time_steps = 1
            if hasattr(safe_model, 'config') and isinstance(safe_model.config, dict):
                safe_model.config['time_steps'] = 1

            model.eval()
            vocab_size = 100
            input_ids = torch.randint(0, vocab_size, (1, 1)).to(self.device)

            for _ in range(10):
                _ = model(input_ids)

            num_runs = 100
            if self.device == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    if hasattr(safe_model, 'reset_net'):
                        safe_model.reset_net()
                    elif hasattr(safe_model, 'model') and hasattr(safe_model.model, 'reset_net'):
                        safe_model.model.reset_net()
                    _ = model(input_ids)

            if self.device == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            avg_latency = ((end_time - start_time) / num_runs) * 1000

            print(f"✅ DONE -> {avg_latency:.2f} ms")

            self.results["tests"][f"efficiency_{model_name}"] = {
                "status": "PASSED",
                "latency_ms": avg_latency
            }
        except Exception as e:
            print(f"❌ FAILED: {e}")


def main():
    suite = BenchmarkSuite()
    models = [("SFormer_Baseline", None), ("BitSpikeMamba_New", None)]
    for name, conf in models:
        suite.run_smoke_test(name, conf)
        suite.run_efficiency_benchmark(name)
    suite.save_report()


if __name__ == "__main__":
    main()
