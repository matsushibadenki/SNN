# ファイルパス: snn_research/cognitive_architecture/explainability.py
# 日本語タイトル: 説明責任 (Explainability) モジュール
# 目的・内容:
#   ROADMAP Phase 3.1 対応。
#   SNNの内部状態（スパイク発火パターン）を解釈可能な自然言語説明に変換する。
#   「なぜその判断をしたのか？」という問いに答えるための基盤。

from typing import Dict, Any, List
import torch
import logging

from .neuro_symbolic_bridge import NeuroSymbolicBridge
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    説明可能性エンジン。
    ニューラル活動と言語的説明の橋渡しを行う。
    """

    def __init__(
        self,
        workspace: GlobalWorkspace,
        bridge: NeuroSymbolicBridge
    ):
        self.workspace = workspace
        self.bridge = bridge
        self.explanation_history: List[str] = []

        logger.info("🗣️ Explainability Engine initialized.")

    def decode_spikes(self, spikes: torch.Tensor, region_name: str) -> List[str]:
        """
        スパイクパターンを意味的なタグや概念にデコードする。
        NeuroSymbolicBridgeを使用して、最も近いシンボルを検索する。
        """
        # ブリッジの逆変換機能を利用 (Pattern -> Symbol)
        # ※ NeuroSymbolicBridgeの実装依存。ここでは簡易的に概念的な説明を返す。

        active_ratio = spikes.float().mean().item()
        explanation = []

        if active_ratio > 0.8:
            explanation.append(f"Highly active {region_name}")
        elif active_ratio < 0.1:
            explanation.append(f"Inactive {region_name}")

        # 登録済みシンボルとの類似度検索（仮実装）
        # symbols = self.bridge.find_symbols_by_pattern(spikes)
        # explanation.extend([s.name for s in symbols])

        return explanation

    def explain_decision(self, decision_context: Dict[str, Any]) -> str:
        """
        意思決定の理由を自然言語で生成する。

        Args:
            decision_context: 決定に至るまでの関連情報 (入力、活性化領域、感情状態など)
        """
        action = decision_context.get("action", "unknown action")
        reasons = decision_context.get("reasons", [])
        emotional_state = decision_context.get("emotion", "neutral")

        explanation = f"I decided to {action} because "

        if reasons:
            explanation += " and ".join(reasons)
        else:
            explanation += "it seemed like the best option based on current intuition."

        if emotional_state != "neutral":
            explanation += f" (Feeling: {emotional_state})"

        self.explanation_history.append(explanation)

        logger.info(f"📝 Generated explanation: {explanation}")
        return explanation

    def generate_introspection_report(self) -> str:
        """
        現在の内部状態に関する内省レポートを生成する。
        """
        # ワークスペースの現在の内容を取得
        conscious_content = self.workspace.get_current_content()

        report = "Introspection Report:\n"
        report += f"Current Focus: {conscious_content.get('type', 'None')}\n"

        return report

    def translate_neural_activity(self, activity_map: Dict[str, torch.Tensor]) -> str:
        """
        脳全体の活動状況を要約説明する。
        """
        active_regions = []
        for region, spikes in activity_map.items():
            if spikes.mean() > 0.5:
                active_regions.append(region)

        if not active_regions:
            return "The brain is currently resting."

        return f"Currently processing information in: {', '.join(active_regions)}."
