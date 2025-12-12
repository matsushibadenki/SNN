# ファイルパス: scripts/run_benchmark_suite.py
# Title: Neuromorphic Benchmark Suite (v16.0) [Fixed]
# Description:
#   ROADMAP v16.0 "安定化と評価基盤" の実装。
#   主要なモデルとタスクに対して、精度・エネルギー・速度のベンチマークを自動実行する。
#   修正: 
#   - run_smoke_test で config_path から正しく設定をロードするように修正。
#   - run_efficiency_benchmark のダミー設定に architecture_type を追加。

import sys
import os
import torch
import logging
import argparse
import time
import json
from typing import Dict, Any, List, cast
from pathlib import Path
import datetime
from omegaconf import OmegaConf

# プロジェクトルートの設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from snn_research.core.snn_core import SNNCore
from snn_research.metrics.energy import EnergyMetrics
from snn_research.io.universal_encoder import UniversalSpikeEncoder

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Benchmark")

class BenchmarkSuite:
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: Dict[str, Any] = {
            "timestamp": str(datetime.datetime.now()),
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "tests": {}
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_smoke_test(self, model_name: str, config_path: str):
        """
        スモークテスト: モデルがエラーなく構築・推論できるか確認。
        """
        logger.info(f"🧪 Running Smoke Test for {model_name}...")
        start_time = time.time()
        status = "FAILED"
        details = ""
        
        try:
            # Config読み込み
            if os.path.exists(config_path):
                conf = OmegaConf.load(config_path)
                # 'model' キー以下にあるか、ルートにあるかを確認して辞書化
                if 'model' in conf:
                    model_config = OmegaConf.to_container(conf.model, resolve=True) # type: ignore
                else:
                    model_config = OmegaConf.to_container(conf, resolve=True)
                
                # コンフィグが正しくロードできたか確認
                if not isinstance(model_config, dict):
                     raise ValueError(f"Config loaded from {config_path} is not a dictionary.")
            else:
                logger.warning(f"Config file not found: {config_path}. Using fallback dummy config.")
                # フォールバック設定 (architecture_typeを追加)
                arch_type = "sformer" if "SFormer" in model_name else "spiking_transformer"
                model_config = {
                    "architecture_type": arch_type,
                    "vocab_size": 100,
                    "d_model": 64,
                    "num_layers": 2,
                    "neuron_config": {"base_threshold": 1.0}
                }
            
            # モデル構築
            # vocab_sizeはconfig内にある場合そちらを優先するロジックもSNNCore側にあると良いが、
            # ここでは引数で渡すかconfig内の値を使う
            vocab_size = model_config.get("vocab_size", 100)
            model = SNNCore(config=cast(Dict[str, Any], model_config), vocab_size=vocab_size).to(self.device)
            model.eval()
            
            # ダミー入力
            input_ids = torch.randint(0, vocab_size, (1, 16)).to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids, return_spikes=True)
                
            status = "PASSED"
            details = "Model built and inference successful."
            
        except Exception as e:
            details = str(e)
            logger.error(f"❌ Smoke Test Failed: {e}")
            import traceback
            traceback.print_exc()
            
        duration = time.time() - start_time
        self.results["tests"][f"smoke_{model_name}"] = {
            "status": status,
            "duration_sec": duration,
            "details": details
        }

    def run_efficiency_benchmark(self, model_name: str, input_shape: tuple = (1, 16)):
        """
        効率ベンチマーク: スパイク率、エネルギー、レイテンシを計測。
        """
        logger.info(f"⚡ Running Efficiency Benchmark for {model_name}...")
        
        try:
            # アーキテクチャタイプの決定 (簡易マッピング)
            if "SFormer" in model_name:
                arch_type = "sformer"
            elif "DSA" in model_name:
                arch_type = "dsa_transformer" # ArchitectureRegistryの実装に合わせる
            else:
                arch_type = "spiking_transformer"

            # モデル構築 (実際は学習済み重みをロードする)
            dummy_config = {
                "architecture_type": arch_type, # 追加
                "vocab_size": 1000,
                "d_model": 128,
                "num_layers": 4,
                "time_steps": 4 # T=4 for SNN
            }
            model = SNNCore(config=dummy_config, vocab_size=1000).to(self.device)
            model.eval()
            
            # 入力データ
            input_ids = torch.randint(0, 1000, input_shape).to(self.device)
            
            # 計測ループ
            num_runs = 100
            total_time = 0.0
            total_spikes = 0.0
            
            # ウォームアップ
            for _ in range(10):
                _ = model(input_ids)
                
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    # スパイク統計をリセット
                    if hasattr(model.model, 'reset_spike_stats'):
                        model.model.reset_spike_stats()
                        
                    _, avg_spike_rate, _ = model(input_ids, return_spikes=True)
                    
                    # スパイク数を集計 (簡易: avg_spike_rate * neurons * time)
                    # 本来は model.get_total_spikes() を使う
                    if hasattr(model.model, 'get_total_spikes'):
                        total_spikes += model.model.get_total_spikes()
                    else:
                        # 推定 (平均発火率 * 時間 * ニューロン数概算)
                        total_spikes += avg_spike_rate.item() * 128 * 16 * 4 
            
            end_time = time.time()
            total_time = end_time - start_time
            
            avg_latency = (total_time / num_runs) * 1000 # ms
            avg_spikes_per_inference = total_spikes / num_runs
            
            # エネルギー推定
            # ニューロン数などは概算
            num_neurons = 128 * 16 * 4 # 仮
            energy_metrics = EnergyMetrics.calculate_energy_consumption(
                total_spikes=avg_spikes_per_inference,
                num_neurons=num_neurons,
                time_steps=dummy_config['time_steps']
            )
            
            ann_comparison = EnergyMetrics.compare_with_ann(
                snn_energy=energy_metrics,
                ann_params=num_neurons * 128 # 仮のパラメータ数
            )
            
            self.results["tests"][f"efficiency_{model_name}"] = {
                "status": "PASSED",
                "latency_ms": avg_latency,
                "spikes_per_inf": avg_spikes_per_inference,
                "energy_joules": energy_metrics,
                "efficiency_gain_vs_ann": ann_comparison['efficiency_gain_percent']
            }
            
            logger.info(f"   -> Latency: {avg_latency:.2f} ms")
            logger.info(f"   -> Energy Gain: {ann_comparison['efficiency_gain_percent']:.1f}%")

        except Exception as e:
            logger.error(f"❌ Efficiency Benchmark Failed: {e}")
            self.results["tests"][f"efficiency_{model_name}"] = {"status": "FAILED", "details": str(e)}

    def save_report(self):
        """結果をJSONとMarkdownで保存"""
        # JSON
        json_path = os.path.join(self.output_dir, "benchmark_latest.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Markdown
        md_path = os.path.join(self.output_dir, "BENCHMARK_REPORT.md")
        with open(md_path, 'w') as f:
            f.write(f"# 📊 Neuromorphic Benchmark Report\n")
            f.write(f"**Date:** {self.results['timestamp']}\n")
            f.write(f"**Device:** {self.results['hardware']}\n\n")
            
            f.write("## 1. Summary\n")
            passed = len([t for t in self.results['tests'].values() if t['status'] == 'PASSED'])
            total = len(self.results['tests'])
            f.write(f"- **Total Tests:** {total}\n")
            f.write(f"- **Passed:** {passed}\n")
            f.write(f"- **Failed:** {total - passed}\n\n")
            
            f.write("## 2. Detailed Results\n")
            f.write("| Test Name | Status | Latency (ms) | Energy Gain (%) | Details |\n")
            f.write("|---|---|---|---|---|\n")
            
            for name, res in self.results['tests'].items():
                latency = f"{res.get('latency_ms', 0):.2f}" if 'latency_ms' in res else "-"
                gain = f"{res.get('efficiency_gain_vs_ann', 0):.1f}" if 'efficiency_gain_vs_ann' in res else "-"
                details = res.get('details', '-')
                status_icon = "✅" if res['status'] == 'PASSED' else "❌"
                
                f.write(f"| {name} | {status_icon} {res['status']} | {latency} | {gain} | {details} |\n")
                
        logger.info(f"📝 Benchmark report saved to {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Run SNN Benchmark Suite")
    parser.add_argument("--mode", type=str, default="all", choices=["smoke", "full"], help="Benchmark mode")
    args = parser.parse_args()
    
    suite = BenchmarkSuite()
    
    # 1. Smoke Tests
    # コンフィグパスは環境に合わせて調整してください
    suite.run_smoke_test("SFormer_T1", "configs/models/phase3_sformer.yaml")
    suite.run_smoke_test("SNN_DSA", "configs/models/dsa_transformer.yaml")
    
    # 2. Efficiency Benchmarks
    if args.mode == "full":
        suite.run_efficiency_benchmark("SFormer_T1")
        suite.run_efficiency_benchmark("SNN_DSA")
        
    suite.save_report()

if __name__ == "__main__":
    main()
