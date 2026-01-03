# ファイルパス: snn_research/cognitive_architecture/basal_ganglia.py
# 日本語タイトル: Basal Ganglia Action Selector v2.1
# 目的・内容:
#   行動選択の中枢。複数の候補（外部提案 + 内部生成）から、
#   情動や報酬予測に基づいて最適な行動を一つ選択する。

from typing import List, Dict, Any, Optional
import torch
from .global_workspace import GlobalWorkspace

class BasalGanglia:
    workspace: GlobalWorkspace

    def __init__(self, workspace: GlobalWorkspace, selection_threshold: float = 0.5, inhibition_strength: float = 0.3):
        self.workspace = workspace
        self.base_threshold = selection_threshold
        self.inhibition_strength = inhibition_strength
        self.selected_action: Optional[Dict[str, Any]] = None
        
        # Workspaceからのブロードキャストも監視するが、
        # メインの行動選択は ArtificialBrain から select_action が呼ばれたときに行う
        self.workspace.subscribe(self.handle_conscious_broadcast)
        print("🧠 大脳基底核モジュールが初期化され、Workspaceを購読しました。")

    def handle_conscious_broadcast(self, source: str, conscious_data: Dict[str, Any]):
        # ここでは直接行動決定せず、内部状態の更新などに留めるのが一般的だが、
        # 簡易実装としてログ出力のみ行う
        # print(f"📬 大脳基底核: '{source}' からの意識的情報を受信。")
        pass

    def _generate_internal_candidates(self) -> List[Dict[str, Any]]:
        """内部的な本能的行動候補を生成する"""
        return [
            {'action': 'investigate_perception', 'value': 0.3}, # デフォルトの探索行動
            {'action': 'reflect_on_emotion', 'value': 0.2},
            {'action': 'ignore', 'value': 0.1}, # 何もしない
        ]

    def _modulate_threshold(self, emotion_context: Optional[Dict[str, float]]) -> float:
        if emotion_context is None:
            return self.base_threshold
        valence = emotion_context.get("valence", 0.0)
        arousal = emotion_context.get("arousal", 0.0)
        # 覚醒度が高いときは閾値を下げて行動しやすくする（衝動的）
        # 不快(Negative Valence)が強いときは、回避行動などを取りやすくする調整も可能
        return max(0.1, min(0.9, self.base_threshold - arousal * 0.2))

    def select_action(
        self, 
        external_candidates: List[Dict[str, Any]], 
        emotion_context: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        外部候補(Reasoning結果など)と内部候補を統合し、行動を選択する。
        """
        self.selected_action = None
        
        # 候補の統合
        internal_candidates = self._generate_internal_candidates()
        all_candidates = external_candidates + internal_candidates
        
        if not all_candidates:
            return None

        current_threshold = self._modulate_threshold(emotion_context)
        
        # 値のテンソル化
        values = torch.tensor([c.get('value', 0.0) for c in all_candidates])
        
        # 最も価値の高い行動を選択 (Winner-Take-All)
        best_idx = torch.argmax(values)
        best_val = values[best_idx].item()
        
        # 閾値判定
        if best_val >= current_threshold:
            self.selected_action = all_candidates[best_idx]
            # print(f"🏆 行動選択: '{self.selected_action.get('action')}' (活性値: {best_val:.2f} >= {current_threshold:.2f})")
            return self.selected_action
        else:
            # print(f"🤔 行動棄却 (Best: {best_val:.2f} < Threshold: {current_threshold:.2f})")
            return None