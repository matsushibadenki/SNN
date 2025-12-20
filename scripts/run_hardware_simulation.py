# ファイルパス: scripts/run_hardware_simulation.py
# Title: ハードウェア・シミュレーション実行スクリプト (Phase 6)
# Description:
#   学習済みのSNNモデルをロードし、イベント駆動型シミュレータ上で実行する。
#   従来の同期型計算（行列演算）と、イベント駆動型計算の「演算量（Ops）」を比較し、
#   SNNのスパース性による効率化を定量的に評価する。

import sys
import os
import torch
import logging
import argparse
from pathlib import Path
from omegaconf import OmegaConf

# プロジェクトルートの設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from snn_research.hardware.event_driven_simulator import EventDrivenSimulator
from snn_research.core.snn_core import SNNCore
from snn_research.distillation.model_registry import SimpleModelRegistry

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("HwSim")

def generate_sparse_input(time_steps: int, input_dim: int, sparsity: float = 0.1) -> torch.Tensor:
    """スパースなダミー入力スパイク列を生成する"""
    # (Time, Features)
    input_data = (torch.rand(time_steps, input_dim) < sparsity).float()
    return input_data

def main():
    parser = argparse.ArgumentParser(description="SNN Event-Driven Hardware Simulation")
    parser.add_argument("--model_config", type=str, default="configs/models/micro.yaml", help="Model config path")
    parser.add_argument("--time_steps", type=int, default=50, help="Simulation duration")
    parser.add_argument("--sparsity", type=float, default=0.05, help="Input spike sparsity (0.0 - 1.0)")
    args = parser.parse_args()

    logger.info("⚡️ SNN Hardware Native Transition Simulation ⚡️")

    # 1. モデルの構築
    logger.info("1. Building SNN Model...")
    if not os.path.exists(args.model_config):
        logger.error(f"Config file not found: {args.model_config}")
        return

    conf = OmegaConf.load(args.model_config)
    # 辞書に変換
    if 'model' in conf:
        model_conf = OmegaConf.to_container(conf.model, resolve=True)
    else:
        model_conf = OmegaConf.to_container(conf, resolve=True)

    # SNNCoreラッパーでモデル生成
    # vocab_size等はダミーでOK
    snn_core = SNNCore(config=model_conf, vocab_size=100)
    model = snn_core.model
    model.eval()

    # 2. シミュレータの初期化
    logger.info("2. Initializing Event-Driven Simulator...")
    try:
        simulator = EventDrivenSimulator(model)
    except Exception as e:
        logger.error(f"Failed to initialize simulator: {e}")
        logger.error("Note: This simulator currently supports simple FeedForward networks with Linear layers.")
        return

    # 3. 入力データの生成
    # モデルの入力次元を推定（最初の重み行列から）
    if simulator.weights:
        input_dim = simulator.weights[0].shape[1]
    else:
        input_dim = 10 # Default
        
    input_spikes = generate_sparse_input(args.time_steps, input_dim, args.sparsity)
    spike_count = input_spikes.sum().item()
    logger.info(f"   - Input Spikes: {int(spike_count)} / {input_spikes.numel()} (Sparsity: {spike_count/input_spikes.numel():.2%})")

    # 4. イベント登録
    simulator.set_input_spikes(input_spikes)

    # 5. 実行
    logger.info("3. Running Simulation...")
    stats = simulator.run(max_time=float(args.time_steps + 10))

    # 6. 比較評価 (同期型 vs イベント駆動型)
    logger.info("\n📊 Performance Analysis (Theoretical)")
    
    # 同期型 (ANN/Synchronous SNN) の計算量推定
    # 全てのニューロンが毎ステップ更新されると仮定
    total_neurons = sum(len(layer) for layer in simulator.layers)
    total_synapses = sum(w.numel() for w in simulator.weights)
    
    # 同期型の総演算数 = (ニューロン数 + シナプス数) * タイムステップ
    # (積和演算を1Opとする)
    sync_ops = total_synapses * args.time_steps
    
    # イベント駆動型の総演算数
    event_ops = stats['total_ops']
    
    reduction_rate = 1.0 - (event_ops / sync_ops) if sync_ops > 0 else 0.0
    speedup = sync_ops / event_ops if event_ops > 0 else float('inf')

    print(f"{'='*40}")
    print(f"Synchronous Ops (Baseline): {sync_ops:,}")
    print(f"Event-Driven Ops (SNN)    : {event_ops:,}")
    print(f"{'-'*40}")
    print(f"📉 Computation Reduction    : {reduction_rate:.2%}")
    print(f"🚀 Theoretical Speedup      : {speedup:.2f}x")
    print(f"{'='*40}")

    if reduction_rate > 0.8:
        logger.info("✅ SUCCESS: Significant efficiency gain demonstrated!")
    else:
        logger.info("⚠️ Note: Efficiency gain is low. Input sparsity or network activity might be too high.")

if __name__ == "__main__":
    main()