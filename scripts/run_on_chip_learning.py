# ファイルパス: scripts/run_on_chip_learning.py
# Title: On-Chip Plasticity デモスクリプト (ログ出力強化版)
# Description:
#   イベント駆動型シミュレータ上で、STDPによる重みのオンライン学習（自己組織化）を実演する。
#   修正: ログが出力されない問題を解決するため、logging設定を強制適用し、
#   print文による即時出力を追加。

import sys
import os
import torch
import torch.nn as nn
import logging
import numpy as np

# プロジェクトルート設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ロギング設定 (強制適用)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("OnChipLearning")

from snn_research.hardware.event_driven_simulator import EventDrivenSimulator
from snn_research.core.neurons import AdaptiveLIFNeuron

def main():
    print(">>> Starting On-Chip Plasticity Demo...", flush=True)
    logger.info("🧠 --- On-Chip Plasticity Demo (Phase 6) ---")
    
    # 1. シンプルなモデルの構築 (入力 10 -> 出力 2)
    print(">>> Building model...", flush=True)
    # 重みはランダム初期化
    model = nn.Sequential(
        nn.Linear(10, 2, bias=False),
        AdaptiveLIFNeuron(features=2, tau_mem=10.0, base_threshold=1.0)
    )
    
    # 重みを小さく初期化 (学習による増強を見やすくする)
    with torch.no_grad():
        model[0].weight.data.uniform_(0.1, 0.2)
    
    initial_weights = model[0].weight.data.clone()
    logger.info(f"Initial Weights (Mean): {initial_weights.mean():.4f}")

    # 2. シミュレータ初期化 (学習有効化)
    print(">>> Initializing simulator...", flush=True)
    simulator = EventDrivenSimulator(
        model, 
        enable_learning=True, 
        learning_rate=0.05, # デモ用に高めに設定
        stdp_window=20.0
    )
    
    # 3. 学習データの生成 (パターンAを繰り返す)
    print(">>> Generating spike patterns...", flush=True)
    # パターンA: 前半のニューロン(0-4)が強く発火
    duration = 50
    input_spikes = torch.zeros(duration, 10)
    
    # パターンを埋め込む
    for t in range(0, duration, 5):
        # ニューロン 0, 2, 4 を発火させる（相関を持たせる）
        input_spikes[t, 0] = 1.0
        input_spikes[t+1, 2] = 1.0
        input_spikes[t+2, 4] = 1.0
        # ノイズ
        if t % 10 == 0:
            input_spikes[t, 9] = 1.0

    # 4. 実行 (On-Chip Learning)
    logger.info("Starting Event-Driven Simulation with STDP...")
    print(">>> Running simulation...", flush=True)
    simulator.set_input_spikes(input_spikes)
    stats = simulator.run(max_time=float(duration + 20))
    
    # 5. 結果確認
    print(">>> Analyzing results...", flush=True)
    final_weights = simulator.weights[0] # Tensor (Simulator内で更新されている)
    weight_diff = final_weights - initial_weights
    
    logger.info(f"Simulation Stats: {stats}")
    logger.info(f"Final Weights (Mean): {final_weights.mean():.4f}")
    
    # パターンAに関与したニューロンへの重みが増加しているか確認
    # Neuron 0, 2, 4 への重みの変化量
    target_indices = [0, 2, 4]
    noise_indices = [1, 3, 5, 6, 7, 8]
    
    # 出力ニューロンごとに確認
    for out_idx in range(2):
        dw_target = weight_diff[out_idx, target_indices].mean().item()
        dw_noise = weight_diff[out_idx, noise_indices].mean().item()
        logger.info(f"Output Neuron {out_idx}:")
        logger.info(f"   - Target Inputs (0,2,4) dW: {dw_target:+.4f}")
        logger.info(f"   - Noise Inputs dW       : {dw_noise:+.4f}")
        
        if dw_target > dw_noise:
            logger.info("   ✅ Correctly potentiated correlated inputs (LTP).")
        else:
            logger.info("   ⚠️ Learning effect weak or LTD dominant.")

    logger.info("🎉 On-Chip Plasticity demo finished.")
    print(">>> Demo finished.", flush=True)

if __name__ == "__main__":
    main()
