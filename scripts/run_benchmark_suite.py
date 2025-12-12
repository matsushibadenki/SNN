# ファイルパス: scripts/run_benchmark_suite.py
# Title: Neuromorphic Benchmark Suite (v16.1 - Debug & MPS Support)
# Description:
#   ベンチマークの実行状況を強制的に標準出力(print)し、サイレント終了を防ぐ。
#   また、Mac (MPS) 環境やCUDA環境を適切に判定する。

import sys
import os
import torch
import logging
import argparse
import time
import json
import traceback
from typing import Dict, Any, List, cast
from pathlib import Path
import datetime
from omegaconf import OmegaConf

# プロジェクトルートの設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ロギング設定 (標準出力へ強制)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("Benchmark")

# 遅延インポートの回避（エラー時に場所を特定しやすくするためここでインポート）
try:
    from snn_research.core.snn_core import SNNCore
    from snn_research.metrics.energy import EnergyMetrics
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

class BenchmarkSuite:
    def __init__(self, output_dir: str = "benchmarks/results"):
        print("⚙️ Initializing Benchmark Suite...")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # デバイス選択 (MPS対応)
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

    def run_smoke_test(self, model_name: str, config_path: str):
        """スモークテスト: モデル構築と推論の確認"""
        print(f"\n🧪 [Smoke Test] {model_name} ... ", end="", flush=True)
        start_time = time.time()
        status = "FAILED"
        details = ""
        
        try:
            # Config読み込み
            if os.path.exists(config_path):
                conf = OmegaConf.load(config_path)
                if 'model' in conf:
                    model_config = OmegaConf.to_container(conf.model, resolve=True) # type: ignore
                else:
                    model_config = OmegaConf.to_container(conf, resolve=True)
            else:
                # フォールバック設定
                print("(Config not found, using dummy) ... ", end="", flush=True)
                arch_type = "sformer" if "SFormer" in model_name else "dsa_transformer"
                model_config = {
                    "architecture_type": arch_type,
                    "vocab_size": 100,
                    "d_model": 64,
                    "num_layers": 2,
                    "neuron_config": {"base_threshold": 1.0}
                }
            
            # モデル構築
            vocab_size = int(model_config.get("vocab_size", 100))
            model = SNNCore(config=cast(Dict[str, Any], model_config), vocab_size=vocab_size).to(self.device)
            model.eval()
            
            # ダミー入力
            input_ids = torch.randint(0, vocab_size, (1, 16)).to(self.device)
            
            # 推論実行
            with torch.no_grad():
                # return_spikes=True を指定して呼び出す
                outputs = model(input_ids, return_spikes=True)
                
            status = "PASSED"
            details = "Model built and inference successful."
            print("✅ PASSED")
            
        except Exception as e:
            details = str(e)
            print(f"❌ FAILED")
            print(f"   Reason: {e}")
            traceback.print_exc()
            
        duration = time.time() - start_time
        self.results["tests"][f"smoke_{model_name}"] = {
            "status": status,
            "duration_sec": duration,
            "details": details
        }

    def run_efficiency_benchmark(self, model_name: str, input_shape: tuple = (1, 16)):
        """効率ベンチマーク"""
        print(f"\n⚡ [Efficiency Test] {model_name} ... ", end="", flush=True)
        
        try:
            # アーキテクチャタイプの決定
            if "SFormer" in model_name:
                arch_type = "sformer"
            elif "DSA" in model_name:
                arch_type = "dsa_transformer"
            else:
                arch_type = "spiking_transformer"

            # モデル構築
            dummy_config = {
                "architecture_type": arch_type,
                "vocab_size": 1000,
                "d_model": 128,
                "num_layers": 4,
                "time_steps": 4,
                "neuron_config": {"base_threshold": 0.5}
            }
            model = SNNCore(config=dummy_config, vocab_size=1000).to(self.device)
            model.eval()
            
            input_ids = torch.randint(0, 1000, input_shape).to(self.device)
            
            # 計測
            num_runs = 50
            total_spikes = 0.0
            
            # Warmup
            for _ in range(5): _ = model(input_ids)
                
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    if hasattr(model.model, 'reset_spike_stats'):
                        model.model.reset_spike_stats()
                        
                    # return_spikes=True で呼び出し
                    # 戻り値の形式: (logits, avg_spike, mem) を想定
                    out = model(input_ids, return_spikes=True)
                    
                    # スパイク数の集計
                    if isinstance(out, tuple) and len(out) >= 2:
                        # 2番目の要素がスパイク情報
                        spike_info = out[1]
                        if isinstance(spike_info, torch.Tensor):
                            # モデル側で計算された平均値や合計値
                            total_spikes += spike_info.item()
                    
                    # バックアップ: モデル内部のカウンタを確認
                    if hasattr(model.model, 'get_total_spikes'):
                        # 直近の実行分を取得できる実装ならここで加算
                        pass 
            
            end_time = time.time()
            avg_latency = ((end_time - start_time) / num_runs) * 1000 # ms
            avg_spikes_per_inference = total_spikes / num_runs
            
            # エネルギー計算 (推定)
            num_neurons = 128 * 16 * 4 # 仮
            energy_metrics = EnergyMetrics.calculate_energy_consumption(
                total_spikes=avg_spikes_per_inference,
                num_neurons=num_neurons,
                time_steps=dummy_config['time_steps']
            )
            
            print(f"✅ DONE")
            print(f"   -> Latency: {avg_latency:.2f} ms")
            print(f"   -> Energy: {energy_metrics:.2e} J")
            
            self.results["tests"][f"efficiency_{model_name}"] = {
                "status": "PASSED",
                "latency_ms": avg_latency,
                "energy_joules": energy_metrics
            }

        except Exception as e:
            print(f"❌ FAILED")
            print(f"   Reason: {e}")
            traceback.print_exc()
            self.results["tests"][f"efficiency_{model_name}"] = {"status": "FAILED", "details": str(e)}

    def save_report(self):
        json_path = os.path.join(self.output_dir, "benchmark_latest.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📝 Report saved to {json_path}")

def main():
    print("🚀 Starting Benchmark Suite...")
    parser = argparse.ArgumentParser(description="Run SNN Benchmark Suite")
    # choices に "all" を追加し、デフォルトで全テストを実行可能に
    parser.add_argument("--mode", type=str, default="all", choices=["smoke", "full", "all"], help="Benchmark mode")
    args = parser.parse_args()
    
    suite = BenchmarkSuite()
    
    # 1. Smoke Tests (Always run)
    suite.run_smoke_test("SFormer_T1", "configs/models/phase3_sformer.yaml")
    suite.run_smoke_test("SNN_DSA", "configs/models/dsa_transformer.yaml")
    
    # 2. Efficiency Benchmarks
    # 修正: 'all' または 'full' の場合に実行
    if args.mode in ["all", "full"]:
        suite.run_efficiency_benchmark("SFormer_T1")
        suite.run_efficiency_benchmark("SNN_DSA")
        
    suite.save_report()
    print("🏁 Benchmark Suite Completed.")

if __name__ == "__main__":
    main()
