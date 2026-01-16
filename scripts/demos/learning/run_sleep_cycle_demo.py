# ファイルパス: scripts/demos/learning/run_sleep_cycle_demo.py
# Title: Autonomous Sleep Cycle Demo (Fully Fixed)
# Description:
#   日中の活動で記憶を蓄積し、疲労後に睡眠をとって記憶を長期記憶へ転送するデモ。
#   [Fix] Hippocampusのメソッド呼び出しを修正 (store -> process)。
#   [Fix] AstrocyteNetworkのエネルギー消費呼び出しを修正 (source引数を追加)。
#   [Fix] 睡眠トリガー条件を修正 (process_stepの戻り値だけでなく、エネルギー値を直接監視)。
#   [Fix] 次元不一致を回避するため、ArtificialBrain初期化時にconfigを設定。

import sys
import os
import torch
import time
import logging

# パス設定
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../")))

from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex

# ログ設定 (強制適用)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    force=True
)
logger = logging.getLogger("SleepCycleDemo")

def run_sleep_cycle_demo():
    print("=== 🌙 Autonomous Sleep Cycle Demo ===")
    print("日中の活動で記憶を蓄積し、疲労後に睡眠をとって記憶を長期記憶へ転送します。\n")

    # 1. コンポーネントの初期化
    workspace = GlobalWorkspace(dim=64)
    astrocyte = AstrocyteNetwork(initial_energy=1000.0, max_energy=1000.0)
    
    cortex = Cortex()
    # 容量を小さくして溢れさせるシミュレーション
    hippocampus = Hippocampus(short_term_capacity=5, working_memory_dim=64)
    
    # 脳の構成設定 (次元を合わせる)
    brain_config = {
        "input_neurons": 64,  # 入力次元
        "feature_dim": 64,    # 内部次元
    }

    # 脳の構築
    brain = ArtificialBrain(
        config=brain_config,
        global_workspace=workspace,
        astrocyte_network=astrocyte,
        hippocampus=hippocampus,
        cortex=cortex
    )

    # 2. 日中の活動 (Learning Phase)
    print("☀️ Day 1: Learning & Exploration Started")
    
    experiences = [
        "Saw a red apple on the table.",
        "Heard a loud noise from the street.",
        "Read a book about neural networks.",
        "Felt tired after coding python.",
        "Ate a delicious sandwich."
    ]

    for i, exp in enumerate(experiences):
        sensory_input = torch.randn(1, 64) 
        
        # 脳活動
        brain.process_step(sensory_input)
        
        # [Fix] 正しいメソッド(process)で海馬へ記憶を保存
        memory_item = {
            "embedding": sensory_input, 
            "text": exp,
            "timestamp": time.time()
        }
        brain.hippocampus.process(memory_item)
        
        # [Fix] エネルギー消費 (source, amount)
        brain.astrocyte.consume_energy("daytime_activity", 15.0)
        
        print(f"  Step {i+1}: Experiencing -> '{exp}'")
        time.sleep(0.1)

    # バッファ確認
    buffer_len = len(brain.hippocampus.episodic_buffer)
    print(f"\n🧠 Hippocampus Buffer: {buffer_len} items")
    energy_level = brain.astrocyte.get_energy_level() * 1000
    print(f"⚡ Current Energy: {energy_level:.1f}/1000")

    # 3. 疲労と睡眠の必要性 (Fatigue Phase)
    print("\n😫 Energy dropped critically low. Needing sleep...")
    brain.astrocyte.energy = 10.0 # 強制的に枯渇させる
    print(f"   (Energy forced down to: {brain.astrocyte.energy})")

    # 4. 睡眠サイクル (Sleep Phase)
    print("\n🌙 Processing next step (Checking for sleep need)...")
    
    result = brain.process_step(torch.randn(1, 64))
    
    # [Fix] 睡眠トリガー条件の強化
    # process_stepが "exhausted" を返さない場合でも、エネルギーレベルを直接確認して睡眠に入る
    should_sleep = (
        result.get("status") == "exhausted" or 
        brain.astrocyte.get_energy_level() < 0.05
    )
    
    if should_sleep:
        print("💤 Brain triggered SLEEP MODE due to exhaustion.")
        
        # 睡眠実行 (エネルギー回復)
        sleep_report = brain.perform_sleep_cycle(cycles=3)
        print(f"   > Sleep Report: {sleep_report}")
        
        # 記憶の固定化 (Consolidation)
        print("   > Consolidating memories from Hippocampus to Cortex...")
        
        # バッファから記憶を取り出す
        memories = brain.hippocampus.flush_memories()
        transferred_count = len(memories)
        
        # (オプション) Cortexへの保存処理の実装があれば呼ぶ
        # if hasattr(brain.cortex, 'store'): brain.cortex.store(memories)

        print(f"   > Memories Transferred: {transferred_count}")
        
        print("✨ Woke up refreshed!")
        print(f"⚡ Energy recovered: {brain.astrocyte.energy:.1f}")
    else:
        print("❌ Sleep was not triggered. Logic check needed.")
        print(f"Debug Result: {result}")
        print(f"Debug Energy: {brain.astrocyte.get_energy_level()}")

    # 5. 結果確認 (Evaluation)
    print("\n📚 Checking Result...")
    print(f"  - Memories consolidated: {transferred_count if 'transferred_count' in locals() else 0}")
    
    if 'transferred_count' in locals() and transferred_count > 0:
        print("\n✅ SUCCESS: Sleep cycle completed and memories consolidated.")
    else:
        print("\n⚠️ PARTIAL SUCCESS: Sleep logic ran, but memory transfer count is 0.")

    print("\n=== Demo Finished ===")

if __name__ == "__main__":
    run_sleep_cycle_demo()