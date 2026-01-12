# ファイルパス: scripts/benchmarks/run_benchmark_suite.py
# Title: Comprehensive Benchmark Suite
# 修正内容: Mypyエラー修正 (SpikingTransformerV2への対応)。

import torch
import time
import logging
import psutil
import os
import sys
from typing import Dict, Any, List, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from snn_research.core.snn_core import SNNCore
# [Mypy Fix] SpikingTransformer -> SpikingTransformerV2
from snn_research.models.transformer.spiking_transformer import SpikingTransformerV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkSuite:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results: Dict[str, Any] = {}
        logger.info(f"🚀 Benchmark Suite Initialized (Device: {self.device})")

    def measure_throughput(self, model: torch.nn.Module, input_shape: tuple, batch_size: int = 32) -> float:
        model.eval()
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                # Transformerの場合、引数が異なる可能性があるためtry-exceptで対応
                try:
                    _ = model(dummy_input)
                except TypeError:
                    # input_idsを想定している場合など
                    dummy_ids = torch.randint(0, 100, (batch_size, input_shape[0])).to(self.device)
                    _ = model(input_ids=dummy_ids)
                
        # Measurement
        start_time = time.time()
        iterations = 50
        for _ in range(iterations):
            with torch.no_grad():
                try:
                    _ = model(dummy_input)
                except TypeError:
                    dummy_ids = torch.randint(0, 100, (batch_size, input_shape[0])).to(self.device)
                    _ = model(input_ids=dummy_ids)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.time()
        total_time = end_time - start_time
        throughput = (batch_size * iterations) / total_time
        return throughput

    def measure_memory(self, model: torch.nn.Module) -> float:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2
        
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            mem_cuda_before = torch.cuda.memory_allocated() / 1024**2
            
        # Dummy forward to allocate activation memory
        dummy_input = torch.randn(1, 64).to(self.device)
        try:
            _ = model(dummy_input)
        except TypeError:
            dummy_ids = torch.randint(0, 100, (1, 64)).to(self.device)
            _ = model(input_ids=dummy_ids)
        
        mem_after = process.memory_info().rss / 1024**2
        cpu_mem = mem_after - mem_before
        
        if self.device == "cuda":
            gpu_mem = (torch.cuda.max_memory_allocated() / 1024**2)
            return gpu_mem
        else:
            return cpu_mem

    def run_core_benchmarks(self):
        logger.info("Running Core SNN Benchmarks...")
        # SNNCore config style
        config = {
            "architecture_type": "hybrid", 
            "hidden_features": 128,
            "in_features": 64,
            "out_features": 10
        }
        model = SNNCore(config=config, vocab_size=10).to(self.device)
        
        tput = self.measure_throughput(model, (64,))
        mem = self.measure_memory(model)
        
        self.results["SNN_Core_Throughput"] = tput
        self.results["SNN_Core_Memory"] = mem
        logger.info(f"SNN Core: {tput:.2f} samples/s, {mem:.2f} MB")

    def run_transformer_benchmarks(self):
        logger.info("Running Spiking Transformer Benchmarks...")
        try:
            # [Mypy Fix] SpikingTransformerV2 初期化
            neuron_config = {
                'tau_mem': 2.0,
                'base_threshold': 1.0,
                'adaptation_strength': 0.1,
                'target_spike_rate': 0.1
            }
            model = SpikingTransformerV2(
                vocab_size=100,
                d_model=64,
                nhead=4,
                num_encoder_layers=2,
                dim_feedforward=128,
                time_steps=8,
                neuron_config=neuron_config,
                img_size=64, # Dummy
                patch_size=16
            ).to(self.device)
            
            # Transformer input: (Batch, Seq) for IDs usually
            tput = self.measure_throughput(model, (64,))
            self.results["Transformer_Throughput"] = tput
            logger.info(f"Transformer: {tput:.2f} samples/s")
            
        except ImportError:
            logger.warning("SpikingTransformer not available, skipping.")
        except Exception as e:
            logger.warning(f"Transformer benchmark failed: {e}")

    def run_scaling_benchmark(self):
        logger.info("Running Scaling Benchmarks...")
        # [Mypy Fix] 型定義を修正して文字列("OOM"等)の代入を許可
        scaling_results: Dict[str, Union[float, str]] = {}
        
        dims = [64, 128, 256, 512]
        for d_model in dims:
            try:
                config = {
                    "architecture_type": "hybrid",
                    "in_features": 64, 
                    "hidden_features": d_model, 
                    "out_features": 10
                }
                model = SNNCore(config=config, vocab_size=10).to(self.device)
                tput = self.measure_throughput(model, (64,))
                scaling_results[f"d{d_model}"] = tput
            except RuntimeError as e:
                if "out of memory" in str(e):
                    scaling_results[f"d{d_model}"] = "OOM"
                else:
                    scaling_results[f"d{d_model}"] = "Error"
            except Exception as e:
                scaling_results[f"d{d_model}"] = str(e)
                
        self.results["Scaling"] = scaling_results
        logger.info(f"Scaling Results: {scaling_results}")

    def save_report(self):
        logger.info("--- Benchmark Report ---")
        for k, v in self.results.items():
            logger.info(f"{k}: {v}")

if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_core_benchmarks()
    suite.run_transformer_benchmarks()
    suite.run_scaling_benchmark()
    suite.save_report()