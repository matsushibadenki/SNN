# ファイルパス: scripts/run_benchmark_suite.py
# Title: SNN Benchmark Suite v2.1 (Robust Fix)
# Description:
#   モデルの推論速度と学習能力を測定するベンチマークスイート。
#   修正 (v2.1):
#   - モデル出力がタプル (logits, auxiliary) の場合のハンドリングを追加。
#   - 出力がシーケンス (B, T, V) か単一ステップ (B, V) かを判定し、
#     損失計算時のターゲット形状を自動調整するように修正。

import sys
import os
import torch
import torch.nn as nn
import logging
import argparse
import time
import json
import datetime
from typing import Dict, Any, cast, Union
from omegaconf import OmegaConf

# プロジェクトルートの設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
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
    from snn_research.metrics.energy import EnergyMetrics
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

class BenchmarkSuite:
    def __init__(self, output_dir: str = "benchmarks/results"):
        print("⚙️ Initializing Benchmark Suite v2.1...")
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
        return {
            "architecture_type": architecture,
            "vocab_size": 100,
            "d_model": 64,
            "num_layers": 2,
            "time_steps": 4,
            "neuron_config": {"base_threshold": 1.0}
        }

    def run_smoke_test(self, model_name: str, config_path: str):
        """スモークテスト: 構築と推論の確認"""
        print(f"\n🧪 [Smoke Test] {model_name} ... ", end="", flush=True)
        
        try:
            # Config読み込み
            if os.path.exists(config_path):
                conf = OmegaConf.load(config_path)
                model_config = OmegaConf.to_container(conf.model if 'model' in conf else conf, resolve=True)
            else:
                arch = "sformer" if "SFormer" in model_name else "dsa_transformer"
                model_config = self._get_dummy_config(arch)
            
            model_config = cast(Dict[str, Any], model_config)
            vocab_size = int(model_config.get("vocab_size", 100))
            
            model = SNNCore(config=model_config, vocab_size=vocab_size).to(self.device)
            model.eval()
            
            input_ids = torch.randint(0, vocab_size, (1, 16)).to(self.device)
            with torch.no_grad():
                _ = model(input_ids)
                
            print("✅ PASSED")
            self.results["tests"][f"smoke_{model_name}"] = {"status": "PASSED"}
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.results["tests"][f"smoke_{model_name}"] = {"status": "FAILED", "error": str(e)}

    def run_training_benchmark(self, model_name: str, steps: int = 50):
        """
        学習ベンチマーク:
        単純な反復パターンを学習させ、Lossの収束と速度を測定する。
        """
        print(f"\n📈 [Training Bench] {model_name} ({steps} steps) ... ", end="", flush=True)
        
        try:
            # モデル準備
            arch = "sformer" if "SFormer" in model_name else "dsa_transformer"
            config = self._get_dummy_config(arch)
            # 学習を容易にするため小さな語彙サイズに設定
            config["vocab_size"] = 10 
            
            model = SNNCore(config=config, vocab_size=10).to(self.device)
            model.train()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
            criterion = nn.CrossEntropyLoss()
            
            # データ生成: [1, 2, 1, 2...]
            batch_size = 8
            seq_len = 16
            x = torch.tensor([[1, 2] * (seq_len // 2) for _ in range(batch_size)]).to(self.device)
            y = torch.tensor([[2, 1] * (seq_len // 2) for _ in range(batch_size)]).to(self.device) # Next token
            
            start_time = time.time()
            initial_loss = 0.0
            final_loss = 0.0
            
            for step in range(steps):
                optimizer.zero_grad()
                outputs = model(x) # (B, T, V) or (B, V) or Tuple
                
                # --- Output Handling Fix ---
                # 1. Tuple処理: (logits, auxiliary_info) の場合は logits を取り出す
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # 2. Shape処理: Sequence output (B, T, V) vs Last-step output (B, V)
                if outputs.dim() == 3:
                    # (B, T, V) -> Flatten to (B*T, V)
                    outputs_flat = outputs.reshape(-1, 10)
                    y_flat = y.reshape(-1)
                    loss = criterion(outputs_flat, y_flat)
                elif outputs.dim() == 2:
                    # (B, V) -> Compare with last target y[:, -1]
                    # モデルが最終ステップのみを出力している場合
                    loss = criterion(outputs, y[:, -1])
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")
                # ---------------------------

                loss.backward()
                optimizer.step()
                
                if step == 0: initial_loss = loss.item()
                final_loss = loss.item()
            
            duration = time.time() - start_time
            steps_per_sec = steps / duration
            
            print(f"✅ DONE")
            print(f"   -> Speed: {steps_per_sec:.1f} steps/s")
            print(f"   -> Loss: {initial_loss:.4f} -> {final_loss:.4f}")
            
            status = "PASSED" if final_loss < initial_loss else "WARNING (No Convergence)"
            
            self.results["tests"][f"train_{model_name}"] = {
                "status": status,
                "steps_per_sec": steps_per_sec,
                "initial_loss": initial_loss,
                "final_loss": final_loss
            }

        except Exception as e:
            print(f"❌ FAILED: {e}")
            # traceback.print_exc() # デバッグ時に有効化
            self.results["tests"][f"train_{model_name}"] = {"status": "FAILED", "error": str(e)}

    def run_efficiency_benchmark(self, model_name: str):
        """効率ベンチマーク"""
        print(f"\n⚡ [Efficiency Test] {model_name} ... ", end="", flush=True)
        try:
            arch = "sformer" if "SFormer" in model_name else "dsa_transformer"
            config = self._get_dummy_config(arch)
            model = SNNCore(config=config, vocab_size=100).to(self.device)
            model.eval()
            
            input_ids = torch.randint(0, 100, (1, 16)).to(self.device)
            
            # Warmup
            for _ in range(5): 
                _ = model(input_ids)
                
            num_runs = 50
            total_spikes = 0.0
            
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    if hasattr(model.model, 'reset_spike_stats'):
                         model.model.reset_spike_stats() # type: ignore
                    
                    out = model(input_ids, return_spikes=True)
                    
                    # スパイク数カウント (Tuple対応)
                    spike_data = None
                    if isinstance(out, tuple) and len(out) >= 2:
                        spike_data = out[1]
                    
                    if spike_data is not None:
                        if isinstance(spike_data, torch.Tensor):
                            total_spikes += spike_data.sum().item()
                        elif isinstance(spike_data, list): # List of tensors
                            total_spikes += sum([s.sum().item() for s in spike_data if isinstance(s, torch.Tensor)])

            end_time = time.time()
            avg_latency = ((end_time - start_time) / num_runs) * 1000
            avg_spikes = total_spikes / num_runs
            
            print(f"✅ DONE")
            print(f"   -> Latency: {avg_latency:.2f} ms")
            
            self.results["tests"][f"efficiency_{model_name}"] = {
                "status": "PASSED",
                "latency_ms": avg_latency,
                "avg_spikes": avg_spikes
            }
        except Exception as e:
            print(f"❌ FAILED: {e}")

    def save_report(self):
        json_path = os.path.join(self.output_dir, "benchmark_latest.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📝 Report saved to {json_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=["smoke", "full", "all"])
    # 互換性引数
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    
    args = parser.parse_args()
    suite = BenchmarkSuite()
    
    # ヘルスチェックからの呼び出し対応
    if args.experiment == "health_check_comparison" or args.tag == "HealthCheck":
         suite.run_smoke_test("HealthCheck_Model", args.model_config or "")
         suite.save_report()
         return

    # 通常実行
    models = [
        ("SFormer_T1", "configs/models/phase3_sformer.yaml"),
        ("SNN_DSA", "configs/models/dsa_transformer.yaml")
    ]
    
    for name, conf in models:
        suite.run_smoke_test(name, conf)
        
        if args.mode in ["all", "full"]:
            suite.run_efficiency_benchmark(name)
            suite.run_training_benchmark(name)
            
    suite.save_report()
    print("🏁 Benchmark Suite Completed.")

if __name__ == "__main__":
    main()
