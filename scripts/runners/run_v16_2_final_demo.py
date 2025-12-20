# ファイルパス: scripts/runners/run_v16_2_final_demo.py
# 日本語タイトル: SNN Roadmap v16.2 Final Integration Demo
# 目的・内容:
#   v16.2の実装完了を証明する統合デモ。
#   1. 未知の環境（Uncertain Inputs）に対するエージェントの反応をシミュレート。
#   2. メタ認知によるSystem 1 -> System 2への切り替えを確認。
#   3. 世界モデルによるシミュレーションと、安全な行動選択を確認。
#   4. アストロサイトによるエネルギー消費の変動を確認。

import sys
import os
import torch
import torch.nn as nn
import logging
import time

# パス設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from snn_research.agent.autonomous_agent import MetaCognitiveAgent
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.models.experimental.world_model_snn import SpikingWorldModel
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.models.transformer.sformer import SFormer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("v16.2_Demo")

class DummyPolicy(nn.Module):
    """System 1 (直感) を模倣する単純なネットワーク"""
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, action_dim)
    def forward(self, x):
        return self.net(x)

def main():
    print("\n" + "="*60)
    print("🧠 SNN v16.2 Integration Demo: Metacognition & World Model")
    print("="*60)
    
    device = "cpu"
    action_dim = 4
    input_dim = 16
    
    # --- 1. コンポーネントの構築 ---
    print("\n🏗️ Building Cognitive Architecture...")
    
    # System 1 Policy (未学習なので出力はランダム＝不確実性が高いはず)
    policy = DummyPolicy(input_dim, action_dim)
    
    # World Model
    wm = SpikingWorldModel(vocab_size=0, d_model=32, action_dim=action_dim, input_dim=input_dim)
    
    # Meta-Cognition
    meta = MetaCognitiveSNN(d_model=action_dim, uncertainty_threshold=0.3)
    
    # Astrocyte (OS)
    astrocyte = AstrocyteNetwork(total_energy_capacity=100.0)
    
    # Agent Assembly
    agent = MetaCognitiveAgent(
        policy_network=policy,
        world_model=wm,
        meta_cognitive=meta,
        astrocyte=astrocyte,
        action_dim=action_dim,
        device=device
    )
    
    print("✅ Agent assembled successfully.")

    # --- 2. シミュレーション実行 ---
    print("\n🏃 Starting Simulation Loop...")
    
    # シナリオ: 10ステップの行動。最初はエネルギー満タン、徐々に疲労。
    for step in range(1, 6):
        print(f"\n--- Step {step} ---")
        
        # 観測データ (ランダム入力)
        obs = torch.randn(input_dim)
        
        # エージェントによる意思決定
        action, info = agent.decide_action(obs)
        
        # 結果表示
        mode = info['mode']
        entropy = info.get('entropy', 0.0)
        energy = astrocyte.current_energy
        
        print(f"   👁️ Observation received.")
        print(f"   🧠 Meta-Cognition: Entropy = {entropy:.4f}")
        
        if mode == "System 2":
            print(f"   💡 System 2 Activated: Running Mental Simulation...")
            # ここでWorld Modelが裏で動いている
        else:
            print(f"   ⚡ System 1 Reflex: Acting on intuition.")
            
        print(f"   🤖 Action Selected: {action}")
        print(f"   🔋 Energy Level: {energy:.1f} (Fatigue: {astrocyte.fatigue_toxin:.1f})")
        
        # 行動結果のフィードバック (ダミー)
        next_obs = torch.randn(input_dim)
        reward = 1.0 if mode == "System 2" else 0.1 # System 2の方が良い結果が出たと仮定
        
        agent.observe_result(obs, action, reward, next_obs)
        
        time.sleep(0.5)

    # --- 3. 疲労によるSystem 2抑制のテスト ---
    print("\n📉 Testing Energy Depletion...")
    # エネルギーを強制的に枯渇させる
    astrocyte.current_energy = 5.0 
    
    print("\n--- Step 6 (Low Energy) ---")
    obs = torch.randn(input_dim)
    action, info = agent.decide_action(obs)
    
    if info.get("resource_denied"):
        print("   🛑 System 2 Request DENIED by Astrocyte (Low Energy).")
        print("   -> Fallback to System 1.")
    else:
        print(f"   Mode: {info['mode']}")

    print("\n🎉 v16.2 Integration Demo Completed.")

if __name__ == "__main__":
    main()