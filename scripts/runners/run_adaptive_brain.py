# scripts/runners/run_adaptive_brain.py
# Title: Adaptive Brain Runner v1.1
# Description: ロギングを強化し、学習後の内部状態（ルール、記憶、シンボル）を検査・表示する機能を追加。

from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.delta_learning import DeltaLearningSystem
from snn_research.cognitive_architecture.neuro_symbolic_bridge import NeuroSymbolicBridge
from snn_research.cognitive_architecture.adaptive_moe import AdaptiveFrankenMoE, Expert
from snn_research.cognitive_architecture.memory_consolidation import HierarchicalMemorySystem
import sys
import os
import numpy as np
import logging
from typing import List, Optional

# プロジェクトルートへのパス設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# ログ設定を強制適用 (force=Trueで既存設定を上書き)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    force=True
)

# 外部ライブラリのログがうるさい場合はレベルを上げる
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ダミーのエキスパートローダー

def load_experts() -> List[Expert]:
    return [
        Expert("visual_expert"),
        Expert("linguistic_expert"),
        Expert("logical_expert"),
        Expert("emotional_expert")
    ]


class AdaptiveBrainSystem:
    def __init__(self):
        print("🚀 Initializing Adaptive Brain System...")

        # 1. 既存コンポーネント
        self.global_workspace = GlobalWorkspace()

        # 2. 適応的MoE
        self.moe = AdaptiveFrankenMoE(load_experts())

        # 3. 階層的記憶システム
        self.memory = HierarchicalMemorySystem()

        # 4. 差分学習システム
        self.delta_learning = DeltaLearningSystem()

        # 5. 神経-記号ブリッジ
        self.neuro_symbolic = NeuroSymbolicBridge(
            snn_network=None, knowledge_graph=None)

        # 制御用
        self.interaction_count = 0
        self.sleep_interval = 5

    def encode_to_spikes(self, text: str) -> np.ndarray:
        """テキスト入力をスパイク活動にエンコード（簡易版）"""
        # 再現性のため、文字列ハッシュをシードにする
        seed = sum(ord(c) for c in text) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(256)  # 256ニューロンの活動

    def process_query(self, user_query: str, user_id: str) -> str:
        """クエリ処理 - 学習機能付きメインループ"""

        print(f"\n👤 User: {user_query}")

        # 1. 入力のエンコード
        spike_input = self.encode_to_spikes(user_query)

        # 2. 適応的エキスパート選択
        experts = self.moe.route_with_learning(
            user_query, user_id, spike_input
        )
        expert_names = [e.name for e in experts]
        print(f"🤖 Activated Experts: {expert_names}")

        # 3. 推論実行 (Global Workspace)
        raw_response = f"Processed '{user_query}' using {expert_names[0]}"

        # 4. 差分学習の適用 (出力の上書き)
        final_response = self.delta_learning.apply_corrections(
            spike_input, raw_response)

        if final_response != raw_response:
            print(
                f"⚡ Delta Correction Applied: {raw_response} -> {final_response}")

        print(f"🧠 Brain: {final_response}")

        # 5. 経験の記憶
        spike_activity = self.global_workspace.broadcast(spike_input)
        if spike_activity is None:
            spike_activity = spike_input

        self.memory.store_experience({
            'query': user_query,
            'response': final_response,
            'experts_used': expert_names
        }, spike_activity)

        # 6. 対話から学習 (Neuro-Symbolic)
        self.neuro_symbolic.learn_from_dialogue(user_query, spike_activity)

        # 7. MoEのフィードバック学習
        # (通常は外部からの正誤判定が必要だが、ここでは「回答できた」ことをポジティブとする)
        for expert in experts:
            self.moe.learn_from_feedback(
                user_query, expert.name, True, spike_input)

        # 8. 定期的な睡眠サイクル
        self.interaction_count += 1
        if self.interaction_count % self.sleep_interval == 0:
            self.sleep()

        return final_response

    def sleep(self):
        """睡眠サイクル - 記憶の統合と最適化"""
        print("\n" + "="*60)
        print("💤 Entering sleep mode (Optimization & Consolidation)...")
        print("="*60)

        # 1. 記憶の固定化（海馬→皮質）
        self.memory.sleep_consolidation(duration_steps=1000)

        # 2. 神経-記号統合
        self.neuro_symbolic.sleep_integration()

        # 3. 差分学習の統合
        self.delta_learning.consolidate_corrections()

        # 4. エキスパート性能の再評価
        self.moe.optimize_routing()

        print("✅ Sleep complete - brain optimized")
        print("="*60 + "\n")

    def provide_feedback(self, query: str, response: str,
                         is_correct: bool, correction: Optional[str] = None):
        """ユーザーフィードバックを受け取り、差分学習に記録"""

        if not is_correct and correction:
            print("📝 User provided feedback: Correction needed.")
            # 修正を記録
            spike_pattern = self.encode_to_spikes(query)
            self.delta_learning.record_correction(
                input_pattern=spike_pattern,
                wrong_output=response,
                correct_output=correction,
                context={'query': query}
            )
            print("✅ Correction recorded for future use.")

    def inspect_state(self):
        """現在の脳の学習状態を検査・表示する"""
        print("\n" + "#"*60)
        print("🔍 BRAIN STATE INSPECTION")
        print("#"*60)

        # 1. MoE ルーティングルール
        print(f"\n[Adaptive MoE Rules] (Total: {len(self.moe.routing_rules)})")
        for h, expert in list(self.moe.routing_rules.items())[:5]:
            print(f"  - QueryHash({h}) -> {expert}")
        print("[Expert Performance]")
        for name, perf in self.moe.expert_performance.items():
            print(f"  - {name}: {perf:.3f}")

        # 2. 記憶システム (海馬)
        print(
            f"\n[Hippocampus Short-term Memory] (Count: {len(self.memory.hippocampus_db)})")
        for k, v in list(self.memory.hippocampus_db.items())[:3]:
            # 内容を少し短縮して表示
            content_preview = (
                v.content[:60] + '..') if len(v.content) > 60 else v.content
            print(f"  - Importancy {v.importance:.2f}: {content_preview}")

        # 3. Neuro-Symbolic Bridge
        print(
            f"\n[Neuro-Symbolic Grounding] (Symbols: {len(self.neuro_symbolic.symbol_to_pattern)})")
        for symbol in list(self.neuro_symbolic.symbol_to_pattern.keys())[:5]:
            print(f"  - Symbol '{symbol}' grounded to neural pattern.")

        # 4. Delta Learning
        print(
            f"\n[Delta Corrections] (Patterns: {len(self.delta_learning.pattern_corrections)})")
        for pattern_key, corrections in self.delta_learning.pattern_corrections.items():
            for c in corrections:
                print(
                    f"  - Correction: '{c['wrong'][:20]}...' -> '{c['correct']}' (Applied: {c['applied_count']} times)")

        print("#"*60 + "\n")


# --- 実行ブロック ---
if __name__ == "__main__":
    brain = AdaptiveBrainSystem()

    # シナリオ実行
    user_id = "user_001"

    # 1. 通常の対話 (MoE学習)
    brain.process_query("What involves visual processing?", user_id)
    brain.process_query("Tell me about logical reasoning.", user_id)

    # 2. 誤りと訂正の学習 (Delta Learning)
    q = "What is the capital of Mars?"
    ans = brain.process_query(q, user_id)
    # ユーザーが訂正
    brain.provide_feedback(q, ans, is_correct=False, correction="Elon City")

    # 3. 再度同じ質問 (差分学習が効くか確認)
    print("\n--- Asking again to test Delta Learning ---")
    brain.process_query(q, user_id)

    # 4. 睡眠サイクルまで対話を続ける (Consolidation & Bridge)
    brain.process_query("Trigger sleep cycle 1", user_id)
    brain.process_query("Trigger sleep cycle 2", user_id)

    # 5. 最終状態の検査
    brain.inspect_state()
