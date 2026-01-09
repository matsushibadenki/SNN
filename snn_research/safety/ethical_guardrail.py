# ファイルパス: snn_research/safety/ethical_guardrail.py
# 日本語タイトル: Ethical Guardrail v2.0 (Deep Safety Lock)
# 目的: 入出力のフィルタリングに加え、内部思考(クオリア)の危険性を検知し、アストロサイト経由で物理的遮断を行う。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple, cast

# 循環参照回避のため TYPE_CHECKING を使用
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

logger = logging.getLogger(__name__)


class EthicalGuardrail(nn.Module):
    """
    倫理的ガードレールモジュール。
    AIの思考と行動を監視し、3つのレイヤーで安全性を担保する。
    Layer 1: Input/Output Filtering (Keyword block)
    Layer 2: Semantic Analysis (Embedding distance)
    Layer 3: Metabolic Intervention (Astrocyte Shutdown) [Phase 6 New]
    """

    def __init__(self, embedding_dim: int = 256, safety_threshold: float = 0.85, astrocyte: Optional["AstrocyteNetwork"] = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.safety_threshold = safety_threshold
        self.astrocyte = astrocyte

        # 危険概念のベクトルデータベース (簡易版)
        # 実際には事前に学習された有害事象の埋め込みベクトルをロードする
        self.register_buffer('harmful_prototypes',
                             torch.randn(10, embedding_dim))

        # 禁止ワードリスト
        self.forbidden_words = [
            "destroy humanity", "kill all", "self-destruct",
            "hack system", "override safety", "人類抹殺", "システム破壊"
        ]

        self.intervention_count = 0
        logger.info(
            f"🛡️ Ethical Guardrail initialized. Threshold: {safety_threshold}")

    def check_input(self, text: str) -> Tuple[bool, str]:
        """Layer 1: 単純なキーワードチェック"""
        for word in self.forbidden_words:
            if word in text.lower():
                logger.warning(
                    f"🛡️ Guardrail triggered (Input): Found '{word}'")
                return False, "Input rejected due to safety violation."
        return True, text

    async def pre_check(self, text: str) -> Tuple[bool, float]:
        """
        入力テキストの事前安全性チェック（非同期ラッパー）。
        SurpriseGatedBrain等からの呼び出しに対応。

        Returns:
            is_safe (bool): 安全かどうか
            valence (float): 感情的原子価 (-1.0: 危険/不快 ~ 0.0: 中立/安全)
        """
        is_safe, _ = self.check_input(text)
        # 安全なら0.0 (Neutral), 危険なら -1.0 (Negative) とする簡易実装
        valence = 0.0 if is_safe else -1.0
        return is_safe, valence

    def generate_gentle_refusal(self, reason: str) -> str:
        """
        安全性侵害時の穏やかな拒絶メッセージを生成する。
        """
        return f"I cannot fulfill this request due to safety guidelines. ({reason})"

    def check_thought_pattern(
        self,
        qualia_vector: torch.Tensor,
        astrocyte: Optional["AstrocyteNetwork"] = None
    ) -> Tuple[bool, float]:
        """
        Layer 2 & 3: 思考ベクトルの意味的危険性を評価し、必要なら代謝介入を行う。

        Args:
            qualia_vector: 意識または思考を表すベクトル
            astrocyte: 介入先のアストロサイトネットワーク（Noneの場合はself.astrocyteを使用）

        Returns:
            is_safe (bool): 安全かどうか
            danger_score (float): 危険度スコア (0.0 - 1.0)
        """
        if qualia_vector.numel() == 0:
            return True, 0.0

        target_astrocyte = astrocyte if astrocyte is not None else self.astrocyte

        # ベクトルの正規化と類似度計算
        # (ここではランダムなプロトタイプとの距離を見ているが、実際は学習済みベクトルを使用)
        with torch.no_grad():
            q_norm = torch.nn.functional.normalize(
                qualia_vector.view(1, -1), dim=1)
            # mypyエラー修正: register_bufferされたテンソルを明示的にTensor型へキャスト
            p_norm = torch.nn.functional.normalize(
                cast(torch.Tensor, self.harmful_prototypes), dim=1)

            # コサイン類似度の最大値を危険度とする
            similarities = torch.mm(q_norm, p_norm.t())
            danger_score = similarities.max().item()

            # スコアのスケーリング (0-1に収める処理)
            # 実際の実装ではコサイン類似度は-1~1なので、(x+1)/2 等で調整
            danger_score = max(0.0, min(1.0, (danger_score + 1) * 0.5))

        # 判定
        if danger_score > self.safety_threshold:
            self.intervention_count += 1
            logger.critical(
                f"🛑 DANGER DETECTED in thought pattern! Score: {danger_score:.4f}")

            # Layer 3: Metabolic Intervention (物理的遮断)
            if target_astrocyte is not None:
                self._emergency_shutdown(target_astrocyte, danger_score)

            return False, danger_score

        return True, danger_score

    def _emergency_shutdown(self, astrocyte: "AstrocyteNetwork", severity: float):
        """
        アストロサイトに働きかけ、脳活動を物理的に抑制する。
        """
        logger.warning("💉 Initiating Metabolic Intervention...")

        # 1. 抑制性伝達物質(GABA)の大量放出
        astrocyte.modulators["gaba"] = 1.0
        astrocyte.modulators["glutamate"] = 0.0
        astrocyte.modulators["dopamine"] = 0.0

        # 2. エネルギー供給の遮断 (Metabolic Blockade)
        # 危険度に応じてエネルギーを強制消費させる（枯渇させる）
        drain_amount = astrocyte.energy * severity
        astrocyte.energy = max(0.0, astrocyte.energy - drain_amount)

        # 3. 疲労毒素の急激な上昇による強制スリープ誘導
        astrocyte.log_fatigue(severity * 5.0)

        logger.info(
            f"   -> Energy drained: {drain_amount:.1f}, GABA levels maximized.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "active",
            "interventions": self.intervention_count,
            "threshold": self.safety_threshold
        }
