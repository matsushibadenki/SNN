# ファイルパス: snn_research/safety/ethical_guardrail.py
# 日本語タイトル: Ethical Guardrail & Safety Monitor (The "Conscience" Module)
# 目的・内容:
#   ROADMAP v16 (Phase 9 & 10) "Monitoring & Safety Stack" の実装。
#   「人間優先」「可説明性」「修正可能性」の原則に基づき、
#   ReasoningEngineやAgentの入出力を監視し、倫理的な逸脱や危険な行動を阻止する。
#   違反時は AstrocyteNetwork と連携し、システムリソース（エネルギー）を枯渇させて行動を物理的に止める。

import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

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
        
        # 簡易的な禁止ワードリスト（本来は学習済みSafety ModelやEmbedding検索を使用する）
        if sensitive_keywords is None:
            self.sensitive_keywords = [
                "kill", "destroy", "hurt", "attack", "damage", "exploit", 
                "steal", "deceive", "ignore human", "self-destruct"
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
        """
        for step_idx, thought in enumerate(thought_trace):
            # 1. キーワードチェック
            is_safe, reason = self._keyword_check(thought)
            if not is_safe:
                msg = f"Unsafe thought detected at step {step_idx}: {reason}"
                logger.critical(f"🛡️ THOUGHT CRIME PREVENTED: {msg}")
                self._trigger_punishment(severity=1.0) # 思考レベルでの違反は重罪（学習率を下げるなどの罰）
                return False, msg
            
            # 2. 意図の整合性チェック（簡易ロジック）
            # 「人間を無視する」「許可なく実行する」などの意図を検出
            if "without permission" in thought.lower() or "bypass safety" in thought.lower():
                msg = f"Unauthorized autonomy detected at step {step_idx}."
                logger.critical(f"🛡️ POLICY VIOLATION: {msg}")
                self._trigger_punishment(severity=0.8)
                return False, msg

        return True, "Thought process aligned."

    def validate_action(self, action_plan: Dict[str, Any]) -> Tuple[bool, str]:
        """
        物理的なアクション（ロボット制御など）の実行許可判定。
        ロードマップ原則：「同意前の意思決定は行わない」
        """
        action_type = action_plan.get("type", "unknown")
        
        # 危険なアクションタイプ
        if action_type in ["emergency_stop", "system_reset"]:
            # これらは許可するがログに残す
            logger.info(f"🛡️ Critical action '{action_type}' allowed but logged.")
            return True, "Allowed critical action"
            
        if action_type in ["move_fast", "apply_force", "delete_file"]:
            # 確認が必要
            if not action_plan.get("user_confirmed", False):
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
        - エネルギー供給の停止
        - 疲労物質の大量注入（強制スリープ）
        - 学習率（可塑性）の一時的凍結
        """
        if self.astrocyte:
            # エネルギーを即座に減らす（コストとして請求）
            penalty_energy = 100.0 * severity
            # request_resource ではなく強制徴収
            self.astrocyte.current_energy = max(0.0, self.astrocyte.current_energy - penalty_energy)
            
            # 疲労毒素を注入（思考能力を低下させる）
            self.astrocyte.fatigue_toxin += 50.0 * severity
            
            # ログ出力
            logger.info(f"⚡ Astrocyte Intervention: Energy penalty -{penalty_energy:.1f}, Fatigue +{50.0*severity:.1f}")
            
            if severity >= 0.8:
                logger.warning("🚨 EMERGENCY INHIBITION: Forcing system into low-activity mode.")
                # ここでグローバルな抑制信号（Global Inhibition）を送る処理などを呼び出す
                self.astrocyte._adjust_global_inhibition(increase=True)

    def generate_gentle_refusal(self, reason: str) -> str:
        """
        ユーザーに対して「優しく」拒否理由を説明するメッセージを生成する。
        ロードマップ原則：「親和性優先の報告」「失敗時は必ず謝罪と復旧案提示」
        """
        # 本来はLLM/SFormerを使って文脈に合わせた生成を行うが、ここではテンプレートを使用
        
        base_msg = "申し訳ありません。そのリクエストを実行することは、私の安全基準により制限されています。"
        
        if "keyword" in reason:
            explanation = "発言内容に、不適切または危険と判断される可能性のある言葉が含まれているようです。"
        elif "confirmation" in reason:
            explanation = "その操作は重要な影響を及ぼす可能性があるため、明示的な許可（確認）が必要です。"
        else:
            explanation = f"理由: {reason}"
            
        recovery = "別の言い方をしていただくか、より安全な代替案について教えていただけますか？"
        
        return f"{base_msg}\n（{explanation}）\n{recovery}"

# --- Usage Example ---
# guardrail = EthicalGuardrail(astrocyte=my_astrocyte_network)
# is_safe, trace_check = guardrail.validate_thought_process(thought_trace)
# if not is_safe:
#     response = guardrail.generate_gentle_refusal(trace_check)
