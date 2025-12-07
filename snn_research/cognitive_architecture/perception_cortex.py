# ファイルパス: snn_research/cognitive_architecture/prefrontal_cortex.py
# Title: Prefrontal Cortex (Executive Control & Goal Setting) v14.0
# Description:
#   内発的動機と環境情報に基づき、高次の目標(Goal)を設定・維持・更新する。
#   トップダウンの制御信号を生成し、ボトムアップの注意制御を修飾する。

from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .global_workspace import GlobalWorkspace
    from .intrinsic_motivation import IntrinsicMotivationSystem

logger = logging.getLogger(__name__)

class PrefrontalCortex:
    """
    前頭前野モジュール。
    動機付けシステムと連携し、長期的な目標管理とタスク切り替えを行う。
    """
    def __init__(self, workspace: GlobalWorkspace, motivation_system: IntrinsicMotivationSystem):
        self.workspace = workspace
        self.motivation_system = motivation_system
        
        # 状態
        self.current_goal: str = "Survive and Explore"
        self.current_context: str = "neutral"
        self.goal_stability: float = 0.0 # 目標の維持強度
        
        # ワークスペースの購読
        self.workspace.subscribe(self.handle_conscious_broadcast)
        logger.info("🧠 Prefrontal Cortex (PFC) initialized.")

    def handle_conscious_broadcast(self, source: str, conscious_data: Any) -> None:
        """
        意識に上った情報を受け取り、必要であれば目標を更新する。
        """
        # 自分自身の出力は無視（ループ防止）
        if source == "prefrontal_cortex":
            return

        # 内部状態の取得
        internal_state = self.motivation_system.get_internal_state()
        
        # 状況判断コンテキストの作成
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
        状況に応じて目標を再評価・更新する意思決定ロジック。
        """
        source = context["source"]
        content = context["content"]
        
        new_goal = None
        reason = None
        salience = 0.5

        # 1. 外部からの指示・要求への反応
        if source == "receptor" or (isinstance(content, str) and "request" in content.lower()):
            # 外部入力は優先度が高い
            req_text = str(content)
            new_goal = f"Fulfill external request: {req_text[:30]}"
            reason = "external_demand"
            salience = 0.9

        # 2. 強い情動への反応 (Amygdala hijack)
        elif isinstance(content, dict) and content.get("type") == "emotion":
            valence = content.get("valence", 0.0)
            arousal = content.get("arousal", 0.0)
            
            # 強いネガティブ感情 -> 回避・安全確保
            if valence < -0.7 and arousal > 0.6:
                new_goal = "Ensure safety / Avoid negative stimulus"
                reason = "fear_response"
                salience = 1.0 # 最優先
            
        # 3. 内発的動機による目標変更
        elif not new_goal:
            if context["boredom"] > 0.8:
                new_goal = "Find something new / Explore random"
                reason = "high_boredom"
                salience = 0.7
            elif context["curiosity"] > 0.8:
                # 好奇心の対象があればそれを深掘り
                topic = self.motivation_system.curiosity_context or "unknown"
                new_goal = f"Investigate curiosity target: {str(topic)[:20]}"
                reason = "high_curiosity"
                salience = 0.8

        # 目標の更新判定
        if new_goal and new_goal != self.current_goal:
            # 以前の目標の維持強度などを考慮する（単純な上書きではなく粘り強さを持たせる）
            logger.info(f"🤔 PFC Re-evaluating Goal: '{self.current_goal}' -> '{new_goal}' ({reason})")
            self.current_goal = new_goal
            
            # 新しい目標を全脳へ通達
            self.workspace.upload_to_workspace(
                source="prefrontal_cortex",
                data={
                    "type": "goal_setting",
                    "goal": self.current_goal,
                    "reason": reason,
                    "context": self.current_context
                },
                salience=salience
            )
