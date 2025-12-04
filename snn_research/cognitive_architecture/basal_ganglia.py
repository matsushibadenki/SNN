# ファイルパス: snn_research/cognitive_architecture/basal_ganglia.py
# (修正: NoneTypeエラー回避)

from typing import List, Dict, Any, Optional
import torch
from .global_workspace import GlobalWorkspace

class BasalGanglia:
    def __init__(self, workspace: GlobalWorkspace, selection_threshold: float = 0.5, inhibition_strength: float = 0.3):
        self.workspace = workspace
        self.base_threshold = selection_threshold
        self.inhibition_strength = inhibition_strength
        self.selected_action: Optional[Dict[str, Any]] = None
        self.workspace.subscribe(self.handle_conscious_broadcast)
        print("🧠 大脳基底核モジュールが初期化され、Workspaceを購読しました。")

    def handle_conscious_broadcast(self, source: str, conscious_data: Dict[str, Any]):
        print(f"📬 大脳基底核: '{source}' からの意識的情報を受信。")
        candidates = [
            {'action': 'investigate_perception', 'value': 0.8},
            {'action': 'reflect_on_emotion', 'value': 0.7},
            {'action': 'ignore', 'value': 0.3},
        ]
        
        # --- ▼ 修正: None安全なアクセス ▼ ---
        emotion_context = None
        if conscious_data is not None and isinstance(conscious_data, dict):
            if conscious_data.get("type") == "emotion":
                emotion_context = conscious_data
        # --- ▲ 修正 ▲ ---
        
        self.select_action(candidates, emotion_context=emotion_context)

    def _modulate_threshold(self, emotion_context: Optional[Dict[str, float]]) -> float:
        if emotion_context is None:
            return self.base_threshold
        valence = emotion_context.get("valence", 0.0)
        arousal = emotion_context.get("arousal", 0.0)
        return max(0.1, min(0.9, self.base_threshold - arousal * 0.2 - valence * arousal * 0.1))

    def select_action(self, action_candidates: List[Dict[str, Any]], emotion_context: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        self.selected_action = None
        if not action_candidates:
            return None
        current_threshold = self._modulate_threshold(emotion_context)
        values = torch.tensor([c.get('value', 0.0) for c in action_candidates])
        best_idx = torch.argmax(values)
        best_val = values[best_idx]
        
        if best_val >= current_threshold:
            self.selected_action = action_candidates[best_idx]
            print(f"🏆 行動選択: '{self.selected_action.get('action')}' (活性値: {best_val:.2f})")
            return self.selected_action
        else:
            print(f"🤔 行動棄却 (閾値: {current_threshold:.2f})")
            return None