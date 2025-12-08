# ファイルパス: scripts/run_on_chip_learning.py
# Title: On-Chip Plasticity デモスクリプト (ノイズ抑制・間隔拡大版)
# Description:
#   イベント駆動型シミュレータ上で、STDPによる重みのオンライン学習（自己組織化）を実演する。
#   修正: ノイズ密度を下げ、パターン間隔を広げて学習を安定化。

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
    
    # 1. モデル構築
    print(">>> Building model...", flush=True)
    model = nn.Sequential(
        nn.Linear(10, 2, bias=False),
        AdaptiveLIFNeuron(features=2, tau_mem=20.0, base_threshold=0.5) 
    )
    
    with torch.no_grad():
        model[0].weight.data.fill_(0.3)
    
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
    
    # 3. 学習データの生成
    print(">>> Generating spike patterns...", flush=True)
    duration = 2000 # さらに時間を延長
    input_spikes = torch.zeros(duration, 10)
    
    pattern_interval = 100 # 間隔を大きく広げる (Traceが完全に消えるのを待つ)
    
    # ノイズの注入 (密度を下げる: 5% -> 0.5%)
    noise_mask = torch.rand(duration, 10) < 0.005
    input_spikes[noise_mask] = 1.0
    
    # ターゲットパターンの埋め込み
    pattern_count = 0
    for t in range(50, duration, pattern_interval):
        input_spikes[t, 0] = 1.0
        input_spikes[t, 2] = 1.0
        input_spikes[t, 4] = 1.0
        # ターゲットと同時にノイズが来ないようにクリア
        input_spikes[t, [1,3,5,6,7,8,9]] = 0.0
        pattern_count += 1
        
    logger.info(f"Pattern injected {pattern_count} times.")

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
    noise_indices = [1, 3, 5, 6, 7, 8, 9]
    
    for out_idx in range(2):
        dw_target = weight_diff[out_idx, target_indices].mean().item()
        dw_noise = weight_diff[out_idx, noise_indices].mean().item()
        
        logger.info(f"Output Neuron {out_idx}:")
        logger.info(f"   - Target Inputs (0,2,4) Mean dW: {dw_target:+.4f}")
        logger.info(f"   - Noise Inputs (Others) Mean dW: {dw_noise:+.4f}")
        
        if dw_target > dw_noise + 0.1:
            logger.info("   ✅ LTP Successful: Target pattern strongly learned!")
        elif dw_target > dw_noise:
             logger.info("   ⚠️ LTP Weak but directional.")
        else:
            logger.info("   ❌ Learning failed.")

    logger.info("🎉 On-Chip Plasticity demo finished.")
    print(">>> Demo finished.", flush=True)

if __name__ == "__main__":
    main()
