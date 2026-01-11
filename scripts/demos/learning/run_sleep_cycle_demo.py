# ファイルパス: scripts/demos/learning/run_sleep_cycle_demo.py
# 日本語タイトル: Sleep Cycle Demo (Autonomous Consolidation)
# 目的: 日中の活動（エピソード記憶）から睡眠時の固定化、夢のリプレイまでの一連の流れを検証する。

from snn_research.utils.brain_debugger import BrainDebugger
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
import sys
import os
import torch
import torch.nn as nn
import logging
import time

# プロジェクトルートにパスを通す
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__)))))


# ロギング設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SleepDemo")


class DummyCoreModel(nn.Module):
    """夢を見るためのダミー脳モデル"""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)  # Dummy

    def forward(self, input_ids=None, input_images=None):
        # 夢の鮮明度(logits)を返すダミー出力
        return torch.randn(1, 10)


def run_demo():
    print("=== 🌙 Autonomous Sleep Cycle Demo ===")
    print("日中の活動で記憶を蓄積し、疲労後に睡眠をとって記憶を長期記憶へ転送します。\n")

    # 1. 脳の初期化
    brain = ArtificialBrain(
        config={"stm_capacity": 10, "curiosity_weight": 0.8})

    # ダミーのコアモデルをセット（夢を見るため）
    brain.set_core_model(DummyCoreModel())

    _ = BrainDebugger(brain)

    # 2. 日中の活動 (Daytime Activity)
    print("\n☀️ Day 1: Learning & Exploration Started")

    experiences = [
        "Saw a red apple on the table.",
        "Heard a loud noise from the street.",
        "Read a book about neural networks.",
        "Felt tired after coding python.",
        "Ate a delicious sandwich."
    ]

    for i, exp in enumerate(experiences):
        print(f"  Step {i+1}: Experiencing -> '{exp}'")

        # 脳に入力 (文字列をそのまま入力としているが、本来はエンコードされたTensor)
        brain.process_step(sensory_input=exp)

        # エネルギー消費シミュレーション (修正: AstrocyteNetwork経由でエネルギーを消費)
        # ArtificialBrain v2.4では max_energy=1000.0 がデフォルト
        brain.astrocyte_network.consume_energy("daytime_activity", 15.0)

        time.sleep(0.5)

    # 現在の短期記憶を確認
    print(
        f"\n🧠 Hippocampus Buffer: {len(brain.hippocampus.episodic_buffer)} items")

    # エネルギー状態の確認 (修正: 正しいAPIを使用)
    current_energy = brain.astrocyte_network.get_energy_level() * \
        1000.0  # Ratio to Absolute
    print(f"⚡ Current Energy: {current_energy:.1f}/1000")

    # 3. 強制的にさらに疲れさせる (Trigger Sleep)
    print("\n😫 Energy dropped critically low. Needing sleep...")

    # 強制的にエネルギーを下げる (AstrocyteNetworkの属性を操作、または大量消費)
    # ここでは疲労物質(fatigue_toxin)を蓄積させ、エネルギーを枯渇させる
    if hasattr(brain.astrocyte_network, 'fatigue_toxin'):
        brain.astrocyte_network.fatigue_toxin = 90.0  # 疲労困憊

    # エネルギーを強制的に下げる（消費メソッドを使用）
    drain_amount = current_energy - 10.0
    if drain_amount > 0:
        brain.astrocyte_network.consume_energy(
            "forced_exhaustion", drain_amount)

    # 確認
    low_energy = brain.astrocyte_network.get_energy_level() * 1000.0
    print(f"   (Energy forced down to: {low_energy:.1f})")

    # 4. 次のステップで自動的に睡眠に入るはず
    print("\n🌙 Processing next step (Should trigger sleep)...")
    result = brain.process_step("Trying to stay awake...")

    # 5. 結果確認
    if result.get("is_sleeping") or result.get("action") == "sleep":
        report = result.get("sleep_report", {})
        print("\n💤 === SLEEP REPORT ===")
        print(
            f"  - Consolidated Memories: {report.get('consolidated_items')} (Moved to Cortex)")
        print(f"  - Dreams Replayed: {report.get('dreams_replayed')}")
        print(
            f"  - Dream Clarity History: {[f'{x:.2f}' for x in report.get('loss_history', [])]}")
        print("✅ Sleep cycle completed successfully.")
    else:
        print("❌ Sleep was not triggered. Check logic.")
        print(f"Debug Result: {result}")

    # 6. 長期記憶の確認
    print("\n📚 Checking Cortex (Long-term Memory)...")
    knowledge = brain.cortex.get_all_knowledge()
    print(f"  - Cortex now contains {len(knowledge)} items.")
    if len(knowledge) > 0:
        print(f"  - Sample knowledge: {knowledge[0][:50]}...")

    print("\n=== Demo Finished ===")


if __name__ == "__main__":
    run_demo()
