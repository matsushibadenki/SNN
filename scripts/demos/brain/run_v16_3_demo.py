# ファイルパス: scripts/runners/run_v16_3_demo.py
# 日本語タイトル: SNN v16.3 Full-Stack Integration Demo (Golden Master)
# 目的・内容:
#   Roadmap v16.3/v17.0 に基づく「反射・直感・熟慮」の階層的意思決定デモ。
#   修正: SmartDummyPolicyがChannel 12に対して「直感的」に反応するように重みを調整。

import sys
import os
import torch
import torch.nn as nn
import logging
import time

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)

from snn_research.agent.autonomous_agent import MetaCognitiveAgent
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.models.experimental.world_model_snn import SpikingWorldModel
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.modules.reflex_module import ReflexModule

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("v16.3_Demo")

class SmartDummyPolicy(nn.Module):
    """
    入力信号の強度に応じて自信（エントロピー）を変化させるダミーポリシー。
    System 1 (直感) の動作を模倣。
    """
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, action_dim)
        # 重みを固定して動作を安定させる
        with torch.no_grad():
            # ベースは弱く（迷い）
            self.net.weight.fill_(0.1)
            
            # 対角成分（0,1,2,3）は強く
            for i in range(action_dim):
                self.net.weight[i, i] = 1.0 
            
            # --- 修正箇所: Channel 12 (Normal State入力) に対する「直感」を植え付ける ---
            # Channel 12 が来たら Action 3 だ！という学習済みパターン
            if input_dim > 12:
                self.net.weight[3, 12] = 2.0 
        
    def forward(self, x):
        # 信号強度(Norm)が高いほど、ロジットを拡大して分布を鋭くする（自信を持たせる）
        signal_strength = torch.norm(x)
        
        if signal_strength < 1.0:
            # 信号微弱 -> 迷う (Entropy High)
            return self.net(x) * 0.5
        else:
            # 信号明瞭 -> 自信満々 (Entropy Low) -> System 1で完結させる
            return self.net(x) * 10.0

def main():
    print("\n" + "="*70)
    print("🧠 SNN Roadmap v16.3 Integration Demo: Reflex, Intuition, Reasoning")
    print("="*70)
    
    device = "cpu"
    action_dim = 4
    input_dim = 16
    
    # --- 1. 脳モジュールの構築 ---
    print("🏗️ Building Artificial Brain Components...")
    
    # Policy: System 1 (Intuition)
    policy = SmartDummyPolicy(input_dim, action_dim)
    
    # World Model: System 2 (Simulation)
    wm = SpikingWorldModel(vocab_size=0, d_model=32, action_dim=action_dim, input_dim=input_dim)
    
    # Meta-Cognition: 閾値0.3 (これを超えるとSystem 2を要請)
    meta = MetaCognitiveSNN(d_model=action_dim, uncertainty_threshold=0.3)
    
    # Reflex: Channel 0-10の入力を監視。閾値3.0を超えると即座に反応。
    reflex = ReflexModule(input_dim=input_dim, action_dim=action_dim, threshold=3.0)
    
    # Astrocyte: エネルギー管理 (デモ用に閾値50.0)
    astrocyte = AstrocyteNetwork(max_energy=100.0, fatigue_threshold=50.0)
    
    agent = MetaCognitiveAgent(
        name="Unit-01",
        policy_network=policy,
        world_model=wm,
        meta_cognitive=meta,
        reflex_module=reflex,
        astrocyte=astrocyte,
        action_dim=action_dim,
        device=device
    )
    print("✅ Complete.\n")

    # --- 2. シナリオ実行 ---
    
    # Scenario 3: DANGER (Reflex)
    # Channel 0 に強い信号 -> Reflex Trigger
    danger_signal = torch.zeros(input_dim)
    danger_signal[0] = 5.0
    
    # Scenario 1: Normal (System 1)
    # Channel 12 に信号を入れる (Reflexの監視範囲外)
    # SmartDummyPolicyの修正により、これに対してAction 3の自信を持つはず
    normal_signal = torch.zeros(input_dim)
    normal_signal[12] = 2.0 
    
    # Scenario 2: Uncertainty (System 2)
    # ノイズのみ -> Reasoning
    uncertain_signal = torch.randn(input_dim) * 0.1

    scenarios = [
        ("Normal State", normal_signal), 
        ("Uncertainty", uncertain_signal),
        ("DANGER!!", danger_signal),
    ]

    print("🏃 Starting Scenarios...")
    
    for i, (name, obs) in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}: {name} ---")
        
        action, info = agent.decide_action(obs)
        
        mode = info.get('mode', 'Unknown')
        energy = astrocyte.current_energy
        entropy = info.get('entropy', 0.0)
        
        print(f"   Input Signal: {obs[0]:.2f} (Channel 0), {obs[12]:.2f} (Channel 12)")
        print(f"   Entropy:      {entropy:.4f}")
        print(f"   Decision:     {mode} -> Action {action}")
        print(f"   Brain Status: Energy={energy:.1f}, Fatigue={astrocyte.fatigue_toxin:.1f}")
        
        # 検証ロジック
        if name == "Normal State" and "System 1" not in mode:
             print("   ⚠️ FAIL: Expected System 1 (Intuition). Entropy is too high.")
        elif name == "Uncertainty" and "System 2" not in mode:
             print("   ⚠️ FAIL: Expected System 2 (Reasoning).")
        elif name == "DANGER!!" and "Reflex" not in mode:
             print("   ❌ FAIL: Expected System 0 (Reflex).")
        else:
             print(f"   ✅ PASS: Correctly selected {mode}")
        
        time.sleep(0.5)

    # --- 3. 疲労限界テスト ---
    print("\n--- Scenario 4: Fatigue Limit (Objective Check) ---")
    print("   forcing heavy fatigue...")
    astrocyte.fatigue_toxin = 60.0 # 閾値(50.0)超過
    
    # 迷う入力を与えてSystem 2を要求させる
    action, info = agent.decide_action(uncertain_signal)
    
    if info.get("resource_denied"):
        print("   ✅ SUCCESS: System 2 Inhibited due to Fatigue.")
        print(f"      Mode: {info['mode']} (Fallback)")
    else:
        print(f"   ❌ FAIL: System 2 ran despite fatigue. Mode: {info.get('mode')}")

    print("\n🎉 v16.3 Demonstration Completed.")

if __name__ == "__main__":
    main()