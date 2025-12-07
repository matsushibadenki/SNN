# ファイルパス: scripts/runners/run_brain_v14.py
# Title: SNN Brain v14.0 Master Simulation
# Description:
#   ロードマップ Phase 5 "Neuro-Symbolic Evolution" の完全デモンストレーション。
#   1. [Awake] SFormerバックボーンによる思考と対話
#   2. [Learning] ユーザーからの知識獲得とGraphRAGへの構造化
#   3. [Sleep] 睡眠サイクルによる記憶の固定化 (Replay -> Synaptic Weight)
#   4. [Evolve] 目標発火率への適応と知識の進化
#   5. [Fatigue] アストロサイトによるリソース枯渇と強制シャットダウン

import sys
import os
import time
import logging
import argparse
from pathlib import Path

# プロジェクトルートの設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.containers import BrainContainer

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger("BrainV14")

def main():
    parser = argparse.ArgumentParser(description="Run SNN Brain v14.0 Simulation")
    parser.add_argument("--config", type=str, default="configs/experiments/brain_v14_config.yaml")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("🧠 SNN Artificial Brain v14.0: Neuro-Symbolic Evolution")
    print("="*60)
    print("   Initializing Neuromorphic OS...")

    # 1. コンテナ初期化
    container = BrainContainer()
    if os.path.exists(args.config):
        container.config.from_yaml(args.config)
    else:
        logger.warning(f"Config file {args.config} not found. Using defaults.")
        # デフォルト設定 (SFormer T=1)
        container.config.from_dict({
            "model": {
                "architecture_type": "sformer",
                "d_model": 128,
                "time_steps": 1,
                "neuron": {"type": "scale_and_fire", "base_threshold": 4.0}
            },
            "training": {
                "biologically_plausible": {
                    "learning_rule": "CAUSAL_TRACE_V2",
                    "neuron": {"type": "lif"}
                }
            }
        })

    # 知識ベースの準備
    rag = container.agent_container.rag_system()
    if not rag.vector_store:
        logger.info("   - Initializing RAG Vector Store...")
        rag.setup_vector_store()

    # 脳の起動
    brain = container.artificial_brain()
    
    # 思考エンジンの確認
    engine_name = brain.thinking_engine.config.get("architecture_type", "unknown")
    print(f"   - Thinking Engine: {engine_name} (Ready)")
    print(f"   - Astrocyte: Energy={brain.astrocyte.current_energy:.1f}")

    # --- シナリオ実行 ---

    # Scene 1: Knowledge Acquisition (対話による学習)
    print("\n🌞 [Phase 1: Knowledge Acquisition]")
    dialogue = [
        "SNN stands for Spiking Neural Network.",
        "SNN uses spikes for energy efficiency.",
        "The brain sleeps to consolidate memory.",
        "Generative replay happens during sleep."
    ]
    
    for txt in dialogue:
        print(f"   👤 Input: '{txt}'")
        result = brain.run_cognitive_cycle(txt)
        
        # 内部状態の表示
        executed = result.get("executed_modules", [])
        print(f"      -> Brain processed via: {executed}")
        if result.get("consciousness"):
            print(f"      -> Consciousness: {result['consciousness']}")
        
        time.sleep(0.5)

    # Scene 2: High Load & Fatigue (思考負荷)
    print("\n🔥 [Phase 2: High Cognitive Load]")
    print("   Simulating complex reasoning tasks to drain energy...")
    
    for i in range(5):
        # 思考エンジンを酷使するタスク
        brain.run_cognitive_cycle(f"Complex reasoning task {i}: Calculate optimal path.")
        energy = brain.astrocyte.current_energy
        fatigue = brain.astrocyte.fatigue_toxin
        print(f"   Task {i+1}: Energy {energy:.1f} | Fatigue {fatigue:.1f}")

    # Scene 3: Sleep & Evolution (睡眠と進化)
    print("\n💤 [Phase 3: Sleep & Consolidation]")
    if brain.state != "SLEEPING":
        print("   Forcing sleep cycle due to roadmap schedule...")
        brain.sleep_cycle()
    
    # Scene 4: Post-Sleep Behavior (進化後の確認)
    print("\n🌞 [Phase 4: Awakening & Evolution Check]")
    
    # 知識の確認
    query = "SNN"
    print(f"   🧠 Checking Long-Term Memory for '{query}':")
    knowledge = brain.cortex.retrieve_knowledge(query)
    for k in knowledge[:3]:
        print(f"      - {k}")
        
    print("\n🎉 Simulation Complete. The Artificial Brain has successfully evolved.")

if __name__ == "__main__":
    main()
