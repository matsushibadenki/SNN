# ファイルパス: snn_research/cognitive_architecture/prefrontal_cortex.py
from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging
from .global_workspace import GlobalWorkspace # 明示的にインポート

if TYPE_CHECKING:
    from .intrinsic_motivation import IntrinsicMotivationSystem

logger = logging.getLogger(__name__)

class PrefrontalCortex:
    # [Fix 3] クラス変数として型アノテーションを追加
    workspace: GlobalWorkspace

    def __init__(self, workspace: GlobalWorkspace, motivation_system: IntrinsicMotivationSystem):
        self.workspace = workspace
        self.motivation_system = motivation_system
        
        self.current_goal: str = "Survive and Explore"
        self.current_context: str = "neutral"
        self.goal_stability: float = 0.0
        self.last_update_reason: str = "initialization"
        
        # [Fix 3] subscribe呼び出し
        self.workspace.subscribe(self.handle_conscious_broadcast)
        logger.info("🧠 Prefrontal Cortex (PFC) initialized.")

    def handle_conscious_broadcast(self, source: str, conscious_data: Any) -> None:
        if source == "prefrontal_cortex":
            return

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
        # (ロジック変更なし)
        source = context["source"]
        content = context["content"]
        
        new_goal: Optional[str] = None
        reason: Optional[str] = None
        salience = 0.5

        if source == "receptor" or (isinstance(content, str) and "request" in content.lower()):
            req_text = str(content)
            new_goal = f"Fulfill external request: {req_text[:50]}"
            reason = "external_demand"
            salience = 0.9
        elif isinstance(content, dict) and content.get("type") == "emotion":
            valence = content.get("valence", 0.0)
            arousal = content.get("arousal", 0.0)
            if valence < -0.7 and arousal > 0.6:
                new_goal = "Ensure safety / Avoid negative stimulus"
                reason = "fear_response"
                salience = 1.0
        elif not new_goal:
            if context["boredom"] > 0.8:
                new_goal = "Find something new / Explore random"
                reason = "high_boredom"
                salience = 0.7
            elif context["curiosity"] > 0.8:
                topic = self.motivation_system.curiosity_context or "unknown"
                new_goal = f"Investigate curiosity target: {str(topic)[:30]}"
                reason = "high_curiosity"
                salience = 0.8

        if new_goal and new_goal != self.current_goal:
            safe_reason: str = reason if reason is not None else "context_change"
            
            logger.info(f"🤔 PFC Re-evaluating Goal: '{self.current_goal}' -> '{new_goal}' ({safe_reason})")
            
            self.current_goal = new_goal
            self.last_update_reason = safe_reason
            
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
        return {
            "goal": self.current_goal,
            "context": self.current_context,
            "reason": self.last_update_reason,
            "stability": self.goal_stability
        }