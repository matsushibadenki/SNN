# ファイルパス: scripts/run_on_chip_learning.py
# Title: On-Chip Plasticity デモスクリプト (パラメータ調整版)
# Description:
#   イベント駆動型シミュレータ上で、STDPによる重みのオンライン学習（自己組織化）を実演する。
#   修正: パターン間隔を広げ、ノイズニューロンも発火するようにして学習効果を明確化。

import sys
import os
import torch
import torch.nn as nn
import logging
import numpy as np

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
    
    # 1. モデル構築 (10入力 -> 2出力)
    print(">>> Building model...", flush=True)
    model = nn.Sequential(
        nn.Linear(10, 2, bias=False),
        AdaptiveLIFNeuron(features=2, tau_mem=10.0, base_threshold=1.0)
    )
    
    # 初期重みを小さく均一に設定
    with torch.no_grad():
        model[0].weight.data.fill_(0.2)
    
    initial_weights = model[0].weight.data.clone()
    logger.info(f"Initial Weights (Uniform): {initial_weights.mean():.4f}")

    # 2. シミュレータ初期化
    print(">>> Initializing simulator...", flush=True)
    simulator = EventDrivenSimulator(
        model, 
        enable_learning=True, 
        learning_rate=0.05, 
        stdp_window=20.0
    )
    
    # 3. 学習データの生成 (パターンAを繰り返す)
    # パターンA: ニューロン 0, 2, 4 が同時発火
    duration = 200 # 時間を長くする
    input_spikes = torch.zeros(duration, 10)
    
    pattern_interval = 40 # 間隔を広げてLTDの影響を減らす
    
    for t in range(0, duration, pattern_interval):
        # ターゲット (同時発火)
        input_spikes[t, 0] = 1.0
        input_spikes[t, 2] = 1.0
        input_spikes[t, 4] = 1.0
        
        # ノイズ (ランダムなタイミング)
        # ターゲットと同期しないようにずらす
        noise_time = t + 10
        if noise_time < duration:
            input_spikes[noise_time, 8] = 1.0 # ニューロン8はノイズ
            input_spikes[noise_time+5, 9] = 1.0 # ニューロン9もノイズ

    # 4. 実行
    logger.info("Starting Event-Driven Simulation with STDP...")
    print(">>> Running simulation...", flush=True)
    simulator.set_input_spikes(input_spikes)
    stats = simulator.run(max_time=float(duration + 20))
    
    # 5. 結果確認
    print(">>> Analyzing results...", flush=True)
    final_weights = simulator.weights[0]
    weight_diff = final_weights - initial_weights
    
    logger.info(f"Simulation Stats: {stats}")
    
    target_indices = [0, 2, 4]
    noise_indices = [8, 9]
    
    for out_idx in range(2):
        dw_target = weight_diff[out_idx, target_indices].mean().item()
        dw_noise = weight_diff[out_idx, noise_indices].mean().item()
        
        logger.info(f"Output Neuron {out_idx}:")
        logger.info(f"   - Target Inputs (0,2,4) dW: {dw_target:+.4f}")
        logger.info(f"   - Noise Inputs (8,9) dW   : {dw_noise:+.4f}")
        
        if dw_target > 0.01 and dw_target > dw_noise:
            logger.info("   ✅ LTP Successful: Target weights increased significantly.")
        else:
            logger.info("   ⚠️ Learning weak or inputs not correlated enough.")

    logger.info("🎉 On-Chip Plasticity demo finished.")
    print(">>> Demo finished.", flush=True)

if __name__ == "__main__":
    main()
