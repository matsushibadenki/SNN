# ファイルパス: snn_research/cognitive_architecture/prefrontal_cortex.py
# (修正: NoneTypeエラー完全回避 + 型チェック強化)

from typing import Dict, Any, Optional
from .global_workspace import GlobalWorkspace
from .intrinsic_motivation import IntrinsicMotivationSystem

class PrefrontalCortex:
    def __init__(self, workspace: GlobalWorkspace, motivation_system: IntrinsicMotivationSystem) -> None:
        self.workspace = workspace
        self.motivation_system = motivation_system
        self.current_goal: str = "Explore and learn"
        self.workspace.subscribe(self.handle_conscious_broadcast)
        print("🧠 前頭前野（実行制御）モジュールが初期化され、Workspaceを購読しました。")

    def handle_conscious_broadcast(self, source: str, conscious_data: Dict[str, Any]):
        if source == "prefrontal_cortex":
            return
        print(f"📬 前頭前野: '{source}' からの意識的情報を受信しました。")
        internal_state = self.motivation_system.get_internal_state()
        system_context = {
            "conscious_content": conscious_data,
            "internal_state": internal_state,
            "external_request": conscious_data if source == "receptor" else None 
        }
        self.decide_goal(system_context)

    def decide_goal(self, system_context: Dict[str, Any]) -> str:
        print("🤔 前頭前野: 次の目標を思考中...")
        
        # --- ▼ 修正: system_context 自体のチェックと各要素の安全な取得 ▼ ---
        if system_context is None:
            system_context = {}

        internal_state = system_context.get("internal_state")
        if internal_state is None:
            internal_state = {}

        conscious_content = system_context.get("conscious_content")
        if conscious_content is None:
            conscious_content = {}
            
        external_request_data = system_context.get("external_request")
        # --- ▲ 修正 ▲ ---

        new_goal = ""
        reason = ""

        if not new_goal and external_request_data:
            request = ""
            if isinstance(external_request_data, dict) and external_request_data.get("type") == "text":
                request = external_request_data.get("content", "")
            elif isinstance(external_request_data, str):
                request = external_request_data
            if request:
                new_goal = f"Fulfill external request: {request}"
                reason = "external_request"

        # --- ▼ 修正: isinstance(conscious_content, dict) を追加 ▼ ---
        if not new_goal and isinstance(conscious_content, dict) and conscious_content.get("type") == "emotion":
            valence = conscious_content.get("valence", 0.0)
            if isinstance(valence, (int, float)) and abs(valence) > 0.7:
                desc = "positive" if valence > 0 else "negative"
                new_goal = f"Respond to strong {desc} emotion"
                reason = f"strong_{desc}_emotion"
        # --- ▲ 修正 ▲ ---

        if not new_goal:
            if internal_state.get("boredom", 0.0) > 0.7:
                new_goal = "Try a new skill"
                reason = "high_boredom"
            elif internal_state.get("curiosity", 0.0) > 0.8:
                new_goal = "Explore new topic"
                reason = "high_curiosity"
            
        if not new_goal:
            new_goal = "Organize knowledge"
            reason = "default_maintenance"

        self.current_goal = new_goal
        print(f"🎯 新目標設定 ({reason}): {self.current_goal}")
        
        self.workspace.upload_to_workspace(
            source="prefrontal_cortex",
            data={"type": "goal_setting", "goal": self.current_goal, "reason": reason},
            salience=0.6
        )
        return self.current_goal