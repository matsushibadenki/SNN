# ファイルパス: scripts/runners/run_world_model_demo.py
# Title: Meta-Cognition & World Model Simulation Demo (Fixed)
# Description:
#   System 1 (直感) と System 2 (熟考/シミュレーション) の切り替えデモ。
#   修正: 引数名の整合性を修正。

import sys
import os
import torch
import logging

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.models.experimental.world_model_snn import SpikingWorldModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("SWM_Demo")

def main():
    print("🌍 --- Spiking World Model & Meta-Cognition Demo ---")
    
    # 1. コンポーネントの初期化
    # System 1 Monitor
    meta_snn = MetaCognitiveSNN(d_model=10) 
    
    # System 2 Simulator (World Model)
    # 修正: vocab_size=0 (連続値入力), latent_dim -> d_model
    swm = SpikingWorldModel(
        vocab_size=0, 
        input_dim=64, 
        action_dim=4, 
        d_model=128
    )
    
    # 2. System 1 のシミュレーション (不確実性の検知)
    print("\n🔹 Phase 1: Meta-Cognitive Monitoring")
    
    # ケースA: 自信がある (Low Entropy)
    logits_confident = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]])
    res_a = meta_snn.monitor_system1_output(logits_confident)
    print(f"   Case A (Confident): Entropy={res_a['entropy']:.4f} -> Trigger System 2? {res_a['trigger_system2']}")
    
    # ケースB: 迷っている (High Entropy)
    logits_uncertain = torch.tensor([[2.0, 2.0, 2.1, 1.9, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    res_b = meta_snn.monitor_system1_output(logits_uncertain)
    print(f"   Case B (Uncertain): Entropy={res_b['entropy']:.4f} -> Trigger System 2? {res_b['trigger_system2']}")
    
    if res_b['trigger_system2']:
        print("   🚨 Uncertainty detected! Switching to Deep Thought Mode (System 2)...")
        
    # 3. 脳内シミュレーション (World Modelによる計画)
    print("\n🔹 Phase 2: Mental Simulation (Planning)")
    print("   Simulating 3 different action plans to resolve uncertainty...")
    
    # 現在の観測 (Dummy Input)
    current_obs = torch.randn(1, 64)
    initial_latent = swm.encode(current_obs)
    
    # 3つの行動プラン (Action Sequences)
    # (Batch, Steps, ActionDim)
    # Plan 1: ランダムな行動
    plan_1 = torch.randn(1, 5, 4) 
    # Plan 2: 待機 (全て0.0に近い)
    plan_2 = torch.zeros(1, 5, 4) + 0.1
    # Plan 3: 特定の行動 (右へ行くなど)
    plan_3 = torch.zeros(1, 5, 4)
    plan_3[:, :, 0] = 2.0 # Strong action on dim 0
    
    plans = {"Random": plan_1, "Wait": plan_2, "Act": plan_3}
    best_plan = None
    max_reward = -float('inf')
    
    for name, actions in plans.items():
        # シミュレーション実行
        sim_result = swm.simulate_trajectory(initial_latent, actions)
        
        # 予測された累積報酬
        total_reward = sim_result["rewards"].sum().item()
        
        print(f"   - Plan '{name}': Expected Reward = {total_reward:.4f}")
        
        if total_reward > max_reward:
            max_reward = total_reward
            best_plan = name
            
    print(f"\n✅ Decision: Selected Plan '{best_plan}' based on mental simulation.")
    
    # 4. フィードバックと学習 (Surprise Detection)
    print("\n🔹 Phase 3: Reality Check (Surprise)")
    
    predicted_next = initial_latent 
    # 実際の結果が予測と大きく異なるとする (Surprise!)
    actual_next = initial_latent + torch.randn_like(initial_latent) * 3.0 
    
    surprise = meta_snn.evaluate_surprise(predicted_next, actual_next)
    print(f"   Surprise Value: {surprise:.4f}")
    
    if surprise > 0.5:
        print("   🧠 Brain detects high surprise! Updating World Model weights (Simulated).")

    print("\n🎉 Demo Completed.")

if __name__ == "__main__":
    main()