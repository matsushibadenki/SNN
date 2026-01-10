# ファイルパス: scripts/benchmarks/run_benchmark_suite.py
# 日本語タイトル: SNN Benchmark Suite v2.3 (Architecture Comparison)
# 目的・内容:
#   モデルの推論速度（Latency）、学習能力（Throughput）、およびメモリ効率を測定する。
#   Transformer(SFormer) と State Space Model(BitSpikeMamba) の比較を行う。

import sys
import os
import torch
import torch.nn as nn
import logging
import argparse
import time
import json
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

# 必要なモジュールのインポート（失敗時は警告のみ）
try:
    from snn_research.core.snn_core import SNNCore
except ImportError:
    SNNCore = None  # type: ignore
    print("⚠️ SNNCore not found. Using mock for structure verification.")

try:
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
except ImportError:
    BitSpikeMamba = None  # type: ignore
    print("⚠️ BitSpikeMamba not found.")


class BenchmarkSuite:
    def __init__(self, output_dir: str = "benchmarks/results"):
        print("⚙️ Initializing Benchmark Suite v2.3...")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # デバイス選択
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
        """ベンチマーク用の共通設定"""
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
        """モデル構築ヘルパー"""
        # 1. Config準備
        if config_path and os.path.exists(config_path):
            conf = OmegaConf.load(config_path)
            model_config = OmegaConf.to_container(
                conf.model if 'model' in conf else conf, resolve=True)
        else:
            # マッピング
            if "Mamba" in model_name:
                arch = "bit_spike_mamba"
            elif "DSA" in model_name:
                arch = "dsa_transformer"
            else:
                arch = "sformer"  # default
            model_config = self._get_dummy_config(arch)

        model_config = cast(Dict[str, Any], model_config)
        vocab_size = int(model_config.get("vocab_size", 100))

        # 2. モデルインスタンス化
        # BitSpikeMambaの直接インスタンス化（SNNCore未対応の場合のフォールバック）
        if model_config.get("architecture_type") == "bit_spike_mamba" and BitSpikeMamba is not None:
            # BitSpikeMamba __init__ args check
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

        # SNNCore経由
        if SNNCore is not None:
            return SNNCore(config=model_config, vocab_size=vocab_size).to(self.device)

        raise ImportError("No suitable model class found.")

    def _measure_model_size(self, model: nn.Module) -> float:
        """パラメータサイズ(MB)を計算"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def run_smoke_test(self, model_name: str, config_path: Optional[str] = None):
        """スモークテスト: 構築と推論の確認"""
        print(f"\n🧪 [Smoke Test] {model_name} ... ", end="", flush=True)

        try:
            model = self._build_model(model_name, config_path)
            model.eval()

            # Size Check
            size_mb = self._measure_model_size(model)

            # Input check
            vocab_size = getattr(model, 'vocab_size', 100)
            if hasattr(model, 'config'):
                vocab_size = model.config.get('vocab_size', 100)

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
            import traceback
            traceback.print_exc()
            self.results["tests"][f"smoke_{model_name}"] = {
                "status": "FAILED", "error": str(e)}

    def run_efficiency_benchmark(self, model_name: str):
        """
        効率ベンチマーク (Latency):
        T=1 (単一ステップ) の推論を行い、リアルタイム応答性能（Reaction Time）を測定。
        """
        print(
            f"\n⚡ [Efficiency Test] {model_name} (T=1 Latency) ... ", end="", flush=True)
        try:
            model = self._build_model(model_name)

            # T=1 強制 (モデルによって設定方法が異なるため属性セット)
            if hasattr(model, 'time_steps'):
                model.time_steps = 1
            if hasattr(model, 'config'):
                model.config['time_steps'] = 1

            model.eval()

            # 入力長: 1 (単一トークン/フレーム)
            vocab_size = 100
            input_ids = torch.randint(0, vocab_size, (1, 1)).to(self.device)

            # Warmup
            for _ in range(10):
                _ = model(input_ids)

            num_runs = 100

            # 同期処理 (CUDA/MPS)
            if self.device == "cuda":
                torch.cuda.synchronize()
            elif self.device == "mps":
                torch.mps.synchronize()

            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    # SNN State Reset check
                    if hasattr(model, 'reset_net'):
                        model.reset_net()  # Direct logic
                    elif hasattr(model, 'model') and hasattr(model.model, 'reset_net'):
                        model.model.reset_net()

                    _ = model(input_ids)

            if self.device == "cuda":
                torch.cuda.synchronize()
            elif self.device == "mps":
                torch.mps.synchronize()

            end_time = time.time()
            avg_latency = ((end_time - start_time) / num_runs) * 1000

            print("✅ DONE")
            print(f"   -> Latency: {avg_latency:.2f} ms / step")

            status = "PASSED"
            # Target: 10ms for Real-time
            if avg_latency > 10.0:
                print("   ⚠️ WARNING: Latency > 10ms (Target not met)")
                status = "WARNING"
            elif avg_latency < 5.0:
                print("   🚀 EXCELLENT: Latency < 5ms (Real-time capable)")

            self.results["tests"][f"efficiency_{model_name}"] = {
                "status": status,
                "latency_ms": avg_latency,
                "note": "Measured with T=1 input"
            }
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

    def save_report(self):
        json_path = os.path.join(self.output_dir, "benchmark_latest.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📝 Report saved to {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full",
                        choices=["smoke", "full"])
    args = parser.parse_args()

    suite = BenchmarkSuite()

    # 比較対象のモデル定義
    # 1. Baseline: 従来のTransformer (Spikformer / SFormer)
    # 2. Target: 新アーキテクチャ (BitSpikeMamba)
    models = [
        ("SFormer_Baseline", None),  # Uses dummy config
        ("BitSpikeMamba_New", None)  # Uses dummy config
    ]

    print("==================================================")
    print("   🏎️  SNN ARCHITECTURE BENCHMARK (Phase 2.3)   ")
    print("==================================================")

    for name, conf in models:
        suite.run_smoke_test(name, conf)

        if args.mode == "full":
            suite.run_efficiency_benchmark(name)
            # Note: Training benchmark is skipped to focus on Latency Gap Analysis

    suite.save_report()
    print("\n🏁 Benchmark Suite Completed.")


if __name__ == "__main__":
    main()
