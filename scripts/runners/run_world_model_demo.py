# ファイルパス: scripts/runners/run_world_model_demo.py
# Title: Meta-Cognition & World Model Simulation Demo
# Description:
#   v16.1 -> v17.0 への架け橋となるデモ。
#   1. System 1 が不確実な入力に直面 (Meta-Cognition Trigger)
#   2. System 2 が起動し、World Model を使って複数の行動計画をシミュレーション
#   3. 最も報酬が高い（安全な）行動を選択して実行

import sys
import os
import torch
import logging
import numpy as np

# パス設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.models.experimental.world_model_snn import SpikingWorldModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("SWM_Demo")

def main():
    print("🌍 --- Spiking World Model & Meta-Cognition Demo ---")
    
    # 1. Initialize Components
    meta_snn = MetaCognitiveSNN(d_model=10) # Logits dimension
    swm = SpikingWorldModel(input_dim=64, action_dim=4, latent_dim=128)
    
    # 2. Simulate System 1 Input (Uncertain Situation)
    print("\n🔹 Phase 1: Meta-Cognitive Monitoring")
    
    # ケースA: 自信がある場合 (Low Entropy)
    logits_confident = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]])
    res_a = meta_snn.monitor_system1_output(logits_confident)
    print(f"   Case A (Confident): Entropy={res_a['entropy']:.4f} -> Trigger System 2? {res_a['trigger_system2']}")
    
    # ケースB: 迷っている場合 (High Entropy - Flat distribution)
    logits_uncertain = torch.tensor([[2.0, 2.0, 2.1, 1.9, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    res_b = meta_snn.monitor_system1_output(logits_uncertain)
    print(f"   Case B (Uncertain): Entropy={res_b['entropy']:.4f} -> Trigger System 2? {res_b['trigger_system2']}")
    
    if res_b['trigger_system2']:
        print("   🚨 Uncertainty detected! Switching to Deep Thought Mode (System 2)...")
        
    # 3. Mental Simulation using World Model
    print("\n🔹 Phase 2: Mental Simulation (Planning)")
    print("   Simulating 3 different action plans to resolve uncertainty...")
    
    # 現在の観測 (Dummy)
    current_obs = torch.randn(1, 64)
    initial_latent = swm.encode(current_obs)
    
    # 3つのプラン (Action Sequences)
    # Plan 1: ランダム
    plan_1 = torch.randn(1, 5, 4) 
    # Plan 2: 特定のパターン
    plan_2 = torch.ones(1, 5, 4) * 0.5
    # Plan 3: 別のパターン
    plan_3 = torch.zeros(1, 5, 4)
    plan_3[:, :, 0] = 1.0 # Action 0 only
    
    plans = {"Random": plan_1, "Stay": plan_2, "GoRight": plan_3}
    best_plan = None
    max_reward = -float('inf')
    
    for name, actions in plans.items():
        # シミュレーション実行
        sim_result = swm.simulate_trajectory(initial_latent, actions)
        
        # 予測された累積報酬
        total_reward = sim_result["rewards"].sum().item()
        final_state_mag = sim_result["final_state"].mean().item()
        
        print(f"   - Plan '{name}': Expected Reward = {total_reward:.4f} (Final State Mag: {final_state_mag:.2f})")
        
        if total_reward > max_reward:
            max_reward = total_reward
            best_plan = name
            
    print(f"\n✅ Decision: Selected Plan '{best_plan}' based on mental simulation.")
    
    # 4. Feedback / Learning Trigger
    print("\n🔹 Phase 3: Reality Check (Surprise)")
    # 実際にPlanを実行して、予測と違った場合 (Surprise)
    predicted_next = initial_latent # 簡易的に変化なしと仮定していたとする
    actual_next = initial_latent + torch.randn_like(initial_latent) * 2.0 # 実際は大きく変化した
    
    surprise = meta_snn.evaluate_surprise(predicted_next, actual_next)
    if surprise > 0.5:
        print("   🧠 Brain detects high surprise! Updating World Model weights (Simulated).")

    print("\n🎉 Demo Completed.")

if __name__ == "__main__":
    main()