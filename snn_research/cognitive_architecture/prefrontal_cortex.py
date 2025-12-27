# ファイルパス: snn_research/cognitive_architecture/prefrontal_cortex.py
# 日本語タイトル: 前頭前野モジュール (循環参照解決版)
# 目的: 実行制御、ゴール設定、および動的なコンテキスト更新を行う。

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

# 循環インポート防止のため、実行時はインポートせず型チェック時のみ有効化
if TYPE_CHECKING:
    from .global_workspace import GlobalWorkspace
    from .intrinsic_motivation import IntrinsicMotivationSystem

logger = logging.getLogger(__name__)

class PrefrontalCortex:
    """
    実行制御（Executive Control）を司る前頭前野モジュール。
    ワークスペースを監視し、内発的動機付けに基づいてゴールを再評価する。
    """
    # 型アノテーションに文字列を使用し、実行時の依存を排除
    workspace: 'GlobalWorkspace'

    def __init__(self, workspace: 'GlobalWorkspace', motivation_system: 'IntrinsicMotivationSystem'):
        """
        Args:
            workspace: GlobalWorkspaceのインスタンス。
            motivation_system: 内発的動機付けシステムのインスタンス。
        """
        self.workspace = workspace
        self.motivation_system = motivation_system
        
        self.current_goal: str = "Survive and Explore"
        self.current_context: str = "neutral"
        self.goal_stability: float = 0.0
        self.last_update_reason: str = "initialization"
        
        # ワークスペースのブロードキャストを購読
        if hasattr(self.workspace, 'subscribe'):
            self.workspace.subscribe(self.handle_conscious_broadcast)
            
        logger.info("🧠 Prefrontal Cortex (PFC) initialized.")

    def handle_conscious_broadcast(self, source: str, conscious_data: Any) -> None:
        """
        ワークスペースからのブロードキャストを受け取り、エグゼクティブ・コントロールを更新する。
        """
        # 自身が発信源の情報は無視
        if source == "prefrontal_cortex":
            return

        # 動機付けシステムから現在の内部状態を取得
        internal_state = self.motivation_system.get_internal_state()
        
        context = {
            "source": source,
            "content": conscious_data,
            "boredom": internal_state.get("boredom", 0.0),
            "curiosity": internal_state.get("curiosity", 0.0),
            "confidence": internal_state.get("confidence", 0.5)
        }
        
        self._update_executive_control(context)

    def _update_executive_control(self, context: Dict[str, Any]):
        """
        知覚や感情に基づいて、現在のゴールや行動指針を決定する。
        """
        source = context["source"]
        content = context["content"]
        
        new_goal: Optional[str] = None
        reason: Optional[str] = None
        salience = 0.5

        # 1. 外部要求（Receptor等）の優先処理
        if source == "receptor" or (isinstance(content, str) and "request" in content.lower()):
            req_text = str(content)
            new_goal = f"Fulfill external request: {req_text[:50]}"
            reason = "external_demand"
            salience = 0.9

        # 2. 感情（恐怖・危機）に基づく生存優先
        elif isinstance(content, dict) and content.get("type") == "emotion":
            valence = content.get("valence", 0.0)
            arousal = content.get("arousal", 0.0)
            if valence < -0.7 and arousal > 0.6:
                new_goal = "Ensure safety / Avoid negative stimulus"
                reason = "fear_response"
                salience = 1.0

        # 3. 内発的動機（退屈・好奇心）に基づく探索
        elif not new_goal:
            if context["boredom"] > 0.8:
                new_goal = "Find something new / Explore random"
                reason = "high_boredom"
                salience = 0.7
            elif context["curiosity"] > 0.8:
                # motivation_systemがcuriosity_context属性を持っていることを想定
                topic = getattr(self.motivation_system, 'curiosity_context', "unknown")
                new_goal = f"Investigate curiosity target: {str(topic)[:30]}"
                reason = "high_curiosity"
                salience = 0.8

        # ゴールが更新された場合の処理
        if new_goal and new_goal != self.current_goal:
            safe_reason: str = reason if reason is not None else "context_change"
            
            logger.info(f"🤔 PFC Re-evaluating Goal: '{self.current_goal}' -> '{new_goal}' ({safe_reason})")
            
            self.current_goal = new_goal
            self.last_update_reason = safe_reason
            
            # ワークスペースへ新しいゴールを提示
            if hasattr(self.workspace, 'upload_to_workspace'):
                self.workspace.upload_to_workspace(
                    source="prefrontal_cortex",
                    data={
                        "type": "goal_setting",
                        "goal": self.current_goal,
                        "reason": safe_reason,
                        "context": self.current_context
                    },
                    salience=salience
                )

    def get_executive_context(self) -> Dict[str, Any]:
        """現在のPFCの状態を取得する"""
        return {
            "goal": self.current_goal,
            "context": self.current_context,
            "reason": self.last_update_reason,
            "stability": self.goal_stability
        }