# ファイルパス: scripts/benchmarks/run_benchmark_suite.py
# 日本語タイトル: SNN Benchmark Suite v2.6 (MPS Memory Support)
# 目的: Apple Silicon (MPS) 環境での正確なメモリプロファイリングと、Phase 2スケーリングテストの強化。

import sys
import os
import torch
import torch.nn as nn
import logging
import time
import datetime
import gc
import psutil
from typing import Dict, Any, cast, Optional, List, Tuple
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

# 外部ライブラリのログ抑制（CuPy警告など）
logging.getLogger("spikingjelly").setLevel(logging.ERROR)

# 依存関係のインポート試行
try:
    from snn_research.core.snn_core import SNNCore
except ImportError:
    SNNCore = None  # type: ignore

try:
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
except ImportError:
    BitSpikeMamba = None  # type: ignore


class BenchmarkSuite:
    def __init__(self, output_dir: str = "workspace/results/benchmarks"):
        print("⚙️ Initializing Benchmark Suite v2.6 (MPS Supported)...")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.device = self._detect_device()
        print(f"   -> Device selected: {self.device}")

        self.results: Dict[str, Any] = {
            "timestamp": str(datetime.datetime.now()),
            "hardware": self.device,
            "tests": {}
        }
        
        # プロセスID取得（CPUメモリ計測用フォールバック）
        self.process = psutil.Process(os.getpid())

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_dummy_config(self, architecture: str, d_model: int = 64) -> Dict[str, Any]:
        """テスト用のダミー設定を生成"""
        base_config = {
            "vocab_size": 100,
            "d_model": d_model,
            "num_layers": 2,
            "time_steps": 8,  # 高速化のためステップ数を調整
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
            # Transformer系
            base_config.update({
                "architecture_type": architecture,
                "nhead": max(2, d_model // 32),
                "dim_feedforward": d_model * 4
            })
        return base_config

    def _build_model(self, model_name: str, config_path: Optional[str] = None, d_model: int = 64) -> nn.Module:
        """モデルを構築してデバイスに転送"""
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
            model_config = self._get_dummy_config(arch, d_model)

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
                time_steps=model_config.get("time_steps", 8),
                neuron_config=model_config["neuron_config"]
            ).to(self.device)

        if SNNCore is not None:
            return SNNCore(config=model_config, vocab_size=vocab_size).to(self.device)

        raise ImportError("No suitable model class found (SNNCore or BitSpikeMamba missing).")

    def _measure_model_size(self, model: nn.Module) -> float:
        """パラメータサイズ(MB)を計測"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024**2

    def _get_current_memory(self) -> float:
        """現在のメモリ使用量(MB)を取得（デバイス依存）"""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1024**2
        elif self.device == "mps":
            # MPSのメモリ計測 (PyTorch 2.0以降推奨)
            try:
                return torch.mps.current_allocated_memory() / 1024**2
            except AttributeError:
                # APIがない場合はRSSメモリの変化で近似
                return self.process.memory_info().rss / 1024**2
        else:
            return self.process.memory_info().rss / 1024**2

    def _measure_peak_memory_during_execution(self, func, *args) -> float:
        """関数の実行中のピークメモリ増加量(MB)を簡易計測"""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        mem_before = self._get_current_memory()
        
        # 実行
        func(*args)
        
        if self.device == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**2
        elif self.device == "mps":
            try:
                # MPSドライバの推奨計測方法
                peak = torch.mps.driver_allocated_memory() / 1024**2
            except AttributeError:
                 # APIがない場合は現在の使用量との差分（不正確だが目安になる）
                mem_after = self._get_current_memory()
                peak = mem_after # 簡易的に現在値をピークとする
        else:
            mem_after = self._get_current_memory()
            peak = mem_after

        # 差分ではなく絶対値、もしくは増加分を返す設計にする
        # ここではモデルロード後の実行時メモリ消費を知りたいので、
        # モデルロード直後のベースラインとの差分が望ましいが、
        # 簡易的に「実行直後のメモリ状態」を返す。
        return peak

    def run_smoke_test(self, model_name: str, config_path: Optional[str] = None):
        """基本動作確認"""
        print(f"\n🧪 [Smoke Test] {model_name} ... ", end="", flush=True)

        try:
            model = self._build_model(model_name, config_path)
            model.eval()
            size_mb = self._measure_model_size(model)

            safe_model = cast(Any, model)
            vocab_size = 100
            if hasattr(safe_model, 'vocab_size'):
                vocab_size = safe_model.vocab_size
            elif hasattr(safe_model, 'config') and isinstance(safe_model.config, dict):
                vocab_size = safe_model.config.get('vocab_size', 100)

            input_ids = torch.randint(0, vocab_size, (1, 16)).to(self.device)

            with torch.no_grad():
                _ = model(input_ids)

            print(f"✅ PASSED (Params: {size_mb:.2f} MB)")
            self.results["tests"][f"smoke_{model_name}"] = {
                "status": "PASSED",
                "param_size_mb": size_mb
            }

        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.results["tests"][f"smoke_{model_name}"] = {
                "status": "FAILED", "error": str(e)}

    def run_efficiency_benchmark(self, model_name: str):
        """推論効率（Latency & Memory）の計測"""
        print(f"⚡ [Latency Test] {model_name} (Batch=1) ... ", end="", flush=True)
        try:
            # モデル構築
            model = self._build_model(model_name)
            safe_model = cast(Any, model)
            
            # 設定の上書き (Latency重視設定)
            if hasattr(safe_model, 'time_steps'):
                safe_model.time_steps = 4 # 少しステップを持たせる
            if hasattr(safe_model, 'config') and isinstance(safe_model.config, dict):
                safe_model.config['time_steps'] = 4

            model.eval()
            vocab_size = 100
            input_ids = torch.randint(0, vocab_size, (1, 1)).to(self.device)

            # ウォームアップ
            for _ in range(5):
                _ = model(input_ids)

            num_runs = 50
            if self.device == "cuda":
                torch.cuda.synchronize()

            # 計測関数
            def run_inference():
                with torch.no_grad():
                    for _ in range(num_runs):
                        if hasattr(safe_model, 'reset_net'):
                            safe_model.reset_net()
                        elif hasattr(safe_model, 'model') and hasattr(safe_model.model, 'reset_net'):
                            safe_model.model.reset_net()
                        _ = model(input_ids)
            
            # メモリ計測しながら実行
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            mem_usage = self._measure_peak_memory_during_execution(run_inference)
            
            if self.device == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            avg_latency = ((end_time - start_time) / num_runs) * 1000

            print(f"✅ {avg_latency:.2f} ms | Est.Mem: {mem_usage:.1f} MB")

            self.results["tests"][f"efficiency_{model_name}"] = {
                "status": "PASSED",
                "latency_ms": avg_latency,
                "memory_mb": mem_usage
            }
        except Exception as e:
            print(f"❌ FAILED: {e}")

    def run_scaling_benchmark(self, model_name: str, scales: List[int] = [64, 128, 256, 512]):
        """スケーラビリティテスト (Phase 2対応)"""
        print(f"\n📈 [Scaling Test] {model_name} checking scales {scales}...")
        
        scaling_results = {}
        
        for d_model in scales:
            print(f"   - d_model={d_model:3d}: ", end="", flush=True)
            
            try:
                # ガベージコレクションを強制してメモリをクリア
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()

                model = self._build_model(model_name, d_model=d_model)
                model.eval()
                
                input_ids = torch.randint(0, 100, (1, 16)).to(self.device)
                
                # ウォームアップ
                for _ in range(3):
                    with torch.no_grad():
                        _ = model(input_ids)
                    
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10): # 高負荷時は回数を減らす
                        _ = model(input_ids)
                        
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    
                elapsed = (time.time() - start_time) / 10 * 1000
                
                # メモリはモデルサイズ + 実行時バッファ
                mem = self._get_current_memory()
                
                print(f"{elapsed:.2f} ms | {mem:.1f} MB")
                scaling_results[f"d{d_model}"] = {"latency": elapsed, "memory": mem}
                
                # 明示的に削除
                del model
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("❌ OOM")
                    scaling_results[f"d{d_model}"] = "OOM"
                else:
                    print(f"❌ Error: {e}")
                    scaling_results[f"d{d_model}"] = "Error"
            except Exception as e:
                print(f"❌ Unexpected: {e}")
                scaling_results[f"d{d_model}"] = str(e)

        self.results["tests"][f"scaling_{model_name}"] = scaling_results

    def save_report(self):
        """レポートの保存"""
        report_path = os.path.join(self.output_dir, "benchmark_report.yaml")
        OmegaConf.save(OmegaConf.create(self.results), report_path)
        print(f"\n📝 Report saved to: {report_path}")


def main():
    suite = BenchmarkSuite()
    
    # Phase 2 重点評価対象
    models = ["SFormer", "BitSpikeMamba"]
    
    for name in models:
        suite.run_smoke_test(name)
        suite.run_efficiency_benchmark(name)
        # MPS/CPU環境では512は重い可能性があるため、try-catch内で実行
        suite.run_scaling_benchmark(name, scales=[64, 128, 256, 512]) 
        
    suite.save_report()


if __name__ == "__main__":
    main()