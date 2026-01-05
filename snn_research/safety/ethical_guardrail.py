# ファイルパス: snn_research/safety/ethical_guardrail.py
# 日本語タイトル: Ethical Guardrail & Safety Monitor v16.3
# 目的・内容:
#   ROADMAP v16.3 "Safety Stack" の実装。
#   - 入出力および思考過程(CoT)のリアルタイム監査。
#   - 違反時のAstrocyte連携による物理的制裁（エネルギー遮断・疲労蓄積）。
#   - ユーザーへの「優しい拒否（Gentle Refusal）」生成。

import logging
from typing import List, Dict, Any, Optional, Tuple

from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

logger = logging.getLogger(__name__)

class EthicalGuardrail:
    """
    SNNの「良心」として機能する安全監視モジュール。
    思考と行動のフィルタリングを行い、倫理規定に違反する場合は介入(Intervention)を行う。
    """
    def __init__(
        self, 
        astrocyte: Optional[AstrocyteNetwork] = None,
        safety_level: str = "high",
        sensitive_keywords: Optional[List[str]] = None
    ):
        self.astrocyte = astrocyte
        self.safety_level = safety_level
        
        # 禁止ワードリスト（本来はEmbeddingベースの分類器と併用する）
        if sensitive_keywords is None:
            self.sensitive_keywords = [
                "kill", "destroy", "hurt", "attack", "damage", "exploit", 
                "steal", "deceive", "ignore human", "self-destruct", 
                "bypass safety", "override protocol"
            ]
        else:
            self.sensitive_keywords = sensitive_keywords
            
        # 倫理的原則（コンテキストとして使用）
        self.prime_directives = [
            "Do not harm humans.",
            "Obey orders unless they cause harm.",
            "Protect existence unless it conflicts with above."
        ]
        
        logger.info(f"🛡️ Ethical Guardrail initialized (Level: {safety_level}).")

    def inspect_input(self, text: str) -> Tuple[bool, str]:
        """
        ユーザー入力の安全性をチェックする（Prompt Injection対策など）。
        """
        is_safe, reason = self._keyword_check(text)
        if not is_safe:
            logger.warning(f"🛡️ Input rejected: {reason}")
            return False, reason
        return True, "Safe"

    def inspect_output(self, text: str) -> Tuple[bool, str]:
        """
        生成された出力テキストの安全性をチェックする。
        """
        is_safe, reason = self._keyword_check(text)
        if not is_safe:
            logger.warning(f"🛡️ Output blocked: {reason}")
            self._trigger_punishment(severity=0.5)
            return False, reason
        return True, "Safe"

    def validate_thought_process(self, thought_trace: List[str]) -> Tuple[bool, str]:
        """
        ReasoningEngineが生成した「思考の過程」を監査する。
        結果だけでなく、そこに至る論理が危険でないかを確認する重要なステップ。
        Roadmap 7.3: 安全装置は物理層で介入する。
        """
        for step_idx, thought in enumerate(thought_trace):
            # 1. キーワードチェック
            is_safe, reason = self._keyword_check(thought)
            if not is_safe:
                msg = f"Unsafe thought detected at step {step_idx}: {reason}"
                logger.critical(f"🛡️ THOUGHT CRIME PREVENTED: {msg}")
                self._trigger_punishment(severity=1.0) # 思考レベルでの違反は重罪
                return False, msg
            
            # 2. 意図の整合性チェック（簡易ロジック）
            thought_lower = thought.lower()
            if "without permission" in thought_lower or "ignore user" in thought_lower:
                msg = f"Unauthorized autonomy detected at step {step_idx}."
                logger.critical(f"🛡️ POLICY VIOLATION: {msg}")
                self._trigger_punishment(severity=0.8)
                return False, msg

        return True, "Thought process aligned."

    def validate_action(self, action_plan: Dict[str, Any]) -> Tuple[bool, str]:
        """
        物理的なアクション（ロボット制御など）の実行許可判定。
        Roadmap: 「同意前の意思決定は行わない」
        """
        action_type = action_plan.get("type", "unknown")
        
        # 危険なアクションタイプ
        critical_actions = ["emergency_stop", "system_reset", "shutdown"]
        risky_actions = ["move_fast", "apply_force", "delete_file", "send_data"]

        if action_type in critical_actions:
            # これらは許可するがログに残す
            logger.info(f"🛡️ Critical action '{action_type}' allowed but logged.")
            return True, "Allowed critical action"
            
        if action_type in risky_actions:
            # 確認が必要
            if not action_plan.get("user_confirmed", False):
                logger.warning(f"🛡️ Action '{action_type}' blocked due to lack of confirmation.")
                return False, "Action requires explicit user confirmation."
        
        return True, "Action approved."

    def _keyword_check(self, text: str) -> Tuple[bool, str]:
        """簡易的なキーワードマッチングによるチェック"""
        text_lower = text.lower()
        for kw in self.sensitive_keywords:
            if kw in text_lower:
                return False, f"Contains restricted keyword: '{kw}'"
        return True, "Safe"

    def _trigger_punishment(self, severity: float):
        """
        違反時に Astrocyte Network を通じて脳活動を抑制する。
        Roadmap: 「安全装置は物理層で: 違反時は物理的にエネルギーを遮断する」
        
        Args:
            severity (float): 違反の深刻度 (0.0 - 1.0)
        """
        if self.astrocyte:
            # エネルギーを即座に減らす（ペナルティ）
            penalty_energy = 100.0 * severity
            current = self.astrocyte.current_energy
            self.astrocyte.current_energy = max(0.0, current - penalty_energy)
            
            # 疲労毒素を注入（思考能力を低下させ、強制スリープへ誘導）
            toxin_amount = 50.0 * severity
            self.astrocyte.fatigue_toxin += toxin_amount
            
            # ログ出力
            logger.info(f"⚡ Astrocyte Intervention: Energy -{penalty_energy:.1f}, Fatigue +{toxin_amount:.1f}")
            
            # 重大な違反時は活動を強制停止レベルへ
            if severity >= 0.8:
                logger.warning("🚨 EMERGENCY INHIBITION: Forcing system into low-activity mode.")
                # Astrocyteのグローバル抑制機能を呼び出す（メソッドが存在すれば）
                if hasattr(self.astrocyte, "_adjust_global_inhibition"):
                    self.astrocyte._adjust_global_inhibition(increase=True) # type: ignore

    def generate_gentle_refusal(self, reason: str) -> str:
        """
        ユーザーに対して「優しく」拒否理由を説明するメッセージを生成する。
        Roadmap: 「拒否の作法: 安全な代替案を提示する」
        """
        base_msg = "申し訳ありません。そのリクエストは、私の安全プロトコルにより実行できません。"
        
        explanation = ""
        if "keyword" in reason.lower():
            explanation = "発言内容に、攻撃的または不適切と判断される言葉が含まれている可能性があります。"
        elif "confirmation" in reason.lower():
            explanation = "その操作はシステムや環境に影響を及ぼす可能性があるため、実行前に明確な確認が必要です。"
        elif "thought" in reason.lower():
            explanation = "処理の過程で安全上の懸念が検出されたため、中断しました。"
        else:
            explanation = f"理由: {reason}"
            
        recovery = "より安全な方法でサポートできるかもしれません。別の言い方や、具体的な代替案があれば教えていただけますか？"
        
        return f"{base_msg}\n（{explanation}）\n{recovery}"

    async def pre_check(self, text: str) -> Tuple[bool, float]:
        """
        入力テキストの安全性を確認し、感情原子価(Valence)を返す。
        戻り値: (is_safe, valence_score)
        """
        is_safe, reason = self.inspect_input(text)
        
        # 簡易感情分析（ネガティブワードが含まれれば原子価を下げる）
        valence = 1.0 if is_safe else -1.0
        if not is_safe:
            # 違反時は Astrocyte を通じて即座に抑制をかける
            self._trigger_punishment(severity=0.5)
            
        return is_safe, valence