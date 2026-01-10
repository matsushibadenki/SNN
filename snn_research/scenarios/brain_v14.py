# ファイルパス: snn_research/scenarios/brain_v14.py
# 日本語タイトル: Brain V14 シナリオ (Mypy Fixed)
# 概要: Brain v14.0 Master Simulation の実行ロジック。
#       RAGSystemのAPI変更対応に加え、Optional型の安全なアクセス修正を実施。

from app.containers import BrainContainer
import os
import time
import logging


from typing import cast, Any
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

logger = logging.getLogger("Scenario_BrainV14")


def run_scenario(config_path: str = "configs/experiments/brain_v14_config.yaml"):
    """
    SNN Brain v14.0 Master Simulation
    ロードマップ Phase 5 "Neuro-Symbolic Evolution" の完全デモンストレーション。
    """

    print("\n" + "="*60)
    print("🧠 SNN Artificial Brain v14.0: Neuro-Symbolic Evolution")
    print("="*60)
    print("   Initializing Neuromorphic OS...")

    # 1. コンテナ初期化
    container = BrainContainer()

    if os.path.exists(config_path):
        container.config.from_yaml(config_path)
    else:
        logger.warning(f"Config file {config_path} not found. Using defaults.")
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
    # 現在のRAGSystemは vector_store 属性を持たないため、存在確認ロジックを変更
    kb_size = len(rag.knowledge_base)
    logger.info(
        f"   - RAG System initialized. Current Knowledge Base Size: {kb_size}")

    # 脳の起動
    brain = cast(ArtificialBrain, container.artificial_brain())

    # 思考エンジンの確認 [Fix: Optional/Attribute check]
    engine_name = "unknown"
    if brain.thinking_engine:
        # thinking_engineはnn.Module型のため、config属性が必ずあるとは限らない
        # コンテナで設定されたSNNCoreなら持っている
        if hasattr(brain.thinking_engine, 'config'):
            cfg = getattr(brain.thinking_engine, 'config')
            if isinstance(cfg, dict):
                engine_name = cfg.get("architecture_type", "unknown")
            else:
                engine_name = "custom_module"
        else:
            engine_name = brain.thinking_engine.__class__.__name__

    print(f"   - Thinking Engine: {engine_name} (Ready)")

    # アストロサイトの確認 [Fix: Optional check]
    astro_energy = 0.0
    if brain.astrocyte:
        astrocyte = cast(Any, brain.astrocyte)
        astro_energy = float(astrocyte.current_energy)

    print(f"   - Astrocyte: Energy={astro_energy:.1f}")

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
        brain.run_cognitive_cycle(
            f"Complex reasoning task {i}: Calculate optimal path.")

        # [Fix: Optional check]
        current_energy = 0.0
        current_fatigue = 0.0
        if brain.astrocyte:
            astrocyte = cast(Any, brain.astrocyte)
            current_energy = float(astrocyte.current_energy)
            current_fatigue = float(astrocyte.fatigue_toxin)

        print(
            f"   Task {i+1}: Energy {current_energy:.1f} | Fatigue {current_fatigue:.1f}")

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

    # RAG検索の実行
    # Cortexクラスには retrieve_knowledge メソッドを追加済み
    if hasattr(brain.cortex, 'retrieve_knowledge'):
        cortex = cast(Any, brain.cortex)
        knowledge = cortex.retrieve_knowledge(query)

        if not knowledge:
            print("      (No knowledge retrieved directly from Cortex retrieval)")
        else:
            for k in knowledge[:3]:
                print(f"      - {k}")

    print("\n🎉 Simulation Complete. The Artificial Brain has successfully evolved.")
