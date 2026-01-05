# ファイルパス: scripts/run_neuro_symbolic_evolution.py
# Title: Neuro-Symbolic Evolution Demo (Phase 5 Completion)
# Description:
#   人工脳が「対話による知識獲得」→「睡眠による記憶固定化」→「知識の再構成」
#   という進化サイクルを自律的に行う様子をシミュレーションする。
#   修正: STDPの必須パラメータ不足によるTypeErrorを修正。

from app.containers import BrainContainer  # E402 fixed
import sys
import os
import time
import logging

# プロジェクトルートの設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("NeuroSymEvo")


def main():
    print("🧠 --- Neuro-Symbolic Evolution Demo (Phase 5) ---")
    print("    Vision: '教えれば賢くなり、眠れば直感になるAI'")

    # 1. コンテナと人工脳の構築
    container = BrainContainer()
    # モデル設定 (Small + Predictive Coding)
    container.config.from_dict({
        "model": {
            "architecture_type": "predictive_coding",
            "d_model": 128,
            "d_state": 64,
            "num_layers": 4,
            "time_steps": 16,
            "neuron": {"type": "lif", "tau_mem": 20.0}
        },
        "training": {
            "biologically_plausible": {
                "neuron": {"type": "lif"},
                # 修正: STDPの初期化に必要なパラメータを追加
                "stdp": {
                    "learning_rate": 0.01,
                    "a_plus": 1.0,
                    "a_minus": 1.0,
                    "tau_trace": 20.0
                }
            }
        }
    })

    # RAG準備
    rag = container.agent_container.rag_system()
    if not rag.vector_store:
        rag.setup_vector_store()

    brain = container.artificial_brain()

    # 2. [AWAKE] 知識の注入 (対話フェーズ)
    print("\n🌞 [Day 1: Learning Phase]")
    inputs = [
        "SNNはスパイキングニューラルネットワークの略です。",
        "SNNは脳の神経回路を模倣しており、省電力です。",
        "睡眠は記憶の定着に重要です。",
        "猫は動物です。",
    ]

    for txt in inputs:
        print(f"  👤 User: {txt}")
        brain.run_cognitive_cycle(txt)
        time.sleep(0.5)

    # 短期記憶(海馬)の状態確認
    wm_size = len(brain.hippocampus.working_memory)
    print(f"  🧠 Hippocampus Working Memory: {wm_size} episodes stored.")

    # 3. [SLEEP] 記憶の固定化 (睡眠フェーズ)
    print("\n💤 [Night 1: Consolidation Phase]")
    print("  ... Brain is initiating sleep cycle to consolidate memories ...")

    # 睡眠サイクル実行 (Hippocampus -> Cortex/GraphRAG -> SNN Replay)
    sleep_report = brain.sleep_cycle()

    print("  ✨ Sleep Report:")
    print(f"     - Phases: {sleep_report['phases']}")
    print(
        f"     - Synaptic Change (Replay Learning): {sleep_report.get('synaptic_change', 0):.4f}")

    # 4. [AWAKE] 知識の検証と訂正 (進化フェーズ)
    print("\n🌞 [Day 2: Application & Correction Phase]")

    # 知識グラフの確認
    print("  🔍 Checking Long-Term Memory (GraphRAG):")
    knowledge = brain.cortex.retrieve_knowledge("SNN")
    print(f"     - Retrieved about 'SNN': {knowledge[:2]}")  # 最初の2件

    # 知識の訂正 (ユーザーによる介入)
    print("\n  🛠️ User Correction: 'SNNは第3世代のAIです' (Correcting knowledge)")
    brain.correct_knowledge(
        concept="SNN",
        correct_info="is the 3rd generation AI",
        reason="Update definition"
    )

    # 修正が反映されたか確認 (睡眠前なのでグラフのみ更新、SNN重みは次回の睡眠で)
    updated_knowledge = brain.cortex.retrieve_knowledge("SNN")
    print(f"     - Updated Knowledge (Graph): {updated_knowledge[-1]}")

    # 5. 強制睡眠 (修正をSNNに焼き付ける)
    print("\n💤 [Night 2: Re-Consolidation Phase]")
    brain.sleep_cycle()
    print("  ✅ Correction consolidated into neural weights.")

    print("\n🎉 Demo Completed: The brain has learned, slept, and evolved.")


if __name__ == "__main__":
    main()
