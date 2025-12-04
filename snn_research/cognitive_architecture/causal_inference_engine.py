# ファイルパス: snn_research/cognitive_architecture/causal_inference_engine.py
# (Phase 2 完了版)
# Title: 因果推論エンジン (Causal Inference Engine)
# Description:
# - 意識の連鎖を観察し、文脈依存の因果関係を推論して知識グラフを構築する。
# - 修正: _get_context_description のダミー実装を廃止。
#   GlobalWorkspace上の情報（特に前頭前野の目標や扁桃体の感情）を参照し、
#   真の「文脈」に基づいた因果学習（Contextual Causal Learning）を実現する。

from typing import Dict, Any, Optional, Tuple
from collections import defaultdict

from .rag_snn import RAGSystem
from .global_workspace import GlobalWorkspace

class CausalInferenceEngine:
    """
    意識の連鎖を観察し、文脈依存の因果関係を推論して知識グラフを構築するエンジン。
    """
    def __init__(
        self,
        rag_system: RAGSystem,
        workspace: GlobalWorkspace,
        inference_threshold: int = 3
    ):
        self.rag_system = rag_system
        self.workspace = workspace
        self.inference_threshold = inference_threshold
        
        self.previous_conscious_info: Optional[Dict[str, Any]] = None
        self.previous_context: Optional[str] = None
        # (Context, Cause, Effect) -> Count
        self.co_occurrence_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        
        self.just_inferred: bool = False
        
        self.workspace.subscribe(self.handle_conscious_broadcast)
        print("🔍 因果推論エンジンが初期化され、Workspaceを購読しました。")

    def reset_inference_flag(self):
        self.just_inferred = False

    def _get_event_description(self, conscious_data: Optional[Dict[str, Any]]) -> Optional[str]:
        """意識に上った生データを抽象的なイベント記述に変換する"""
        if not conscious_data:
            return None
        
        # 辞書型の場合の処理
        if isinstance(conscious_data, dict):
            event_type = conscious_data.get("type")
            
            if event_type == "emotion":
                valence = conscious_data.get("valence", 0.0)
                if valence < -0.5: return "strong_negative_emotion"
                if valence > 0.5: return "strong_positive_emotion"
                return "neutral_emotion"
                
            elif event_type == "perception" or event_type == "visual_perception":
                # 知覚内容は詳細すぎるため、一般的なラベルにするか、検出物体があればそれを使う
                detected = conscious_data.get("detected_objects")
                if detected:
                    labels = [d['label'] for d in detected]
                    return f"perception_of_{'_'.join(labels[:2])}"
                return "novel_perception"
                
            elif event_type == "goal_setting":
                return f"goal_set_{conscious_data.get('reason', 'unknown')}"
                
            elif event_type == "causal_credit":
                return None # クレジット信号自体は因果のノードにしない

            # アクションの場合
            if 'action' in conscious_data:
                return f"action_{conscious_data['action']}"
        
        # 文字列の場合
        elif isinstance(conscious_data, str):
            if conscious_data.startswith("Fulfill external request"):
                return "external_request_received"
            return "general_observation"

        return "unknown_event"

    def _get_context_description(self) -> str:
        """
        【Phase 2 完全化】
        現在の脳内文脈を取得する。ダミーではなく、実際のWorkspaceの状態を参照する。
        優先順位:
        1. 前頭前野の現在の目標 (Active Goal)
        2. 扁桃体の現在の感情 (Emotional State)
        3. デフォルト
        """
        # 1. 前頭前野からの情報を確認
        pfc_info = self.workspace.get_information("prefrontal_cortex")
        if pfc_info and isinstance(pfc_info, dict) and "goal" in pfc_info:
            # 目標を文脈とする (例: "context_Fulfill_external_request")
            goal_short = str(pfc_info['goal'])[:20].replace(" ", "_")
            return f"ctx_goal_{goal_short}"

        # 2. 感情状態を確認
        amygdala_info = self.workspace.get_information("amygdala")
        if amygdala_info and isinstance(amygdala_info, dict):
            valence = amygdala_info.get("valence", 0.0)
            if valence < -0.3: return "ctx_negative_mood"
            if valence > 0.3: return "ctx_positive_mood"

        return "ctx_neutral"

    def handle_conscious_broadcast(self, source: str, conscious_data: Dict[str, Any]):
        """
        意識に上った情報の連鎖と、その時の文脈を観察し、因果関係を推論する。
        """
        # クレジット信号自体は因果推論の対象外（ループ防止）
        if isinstance(conscious_data, dict) and conscious_data.get("type") == "causal_credit":
            return

        current_event = self._get_event_description(conscious_data)
        previous_event = self._get_event_description(self.previous_conscious_info)
        
        # 動的に文脈を取得
        current_context = self._get_context_description()

        if previous_event and current_event and self.previous_context:
            # 文脈が継続している場合のみ因果を推論
            if current_context == self.previous_context:
                event_tuple = (self.previous_context, previous_event, current_event)
                self.co_occurrence_counts[event_tuple] += 1
                
                count = self.co_occurrence_counts[event_tuple]
                # デバッグ出力
                # print(f"  - 因果推論観測: [{self.previous_context}] {previous_event} -> {current_event} (回数: {count})")

                if count == self.inference_threshold:
                    print(f"  - 🔥 因果関係を推論・確定！: [{self.previous_context}] {previous_event} -> {current_event}")
                    
                    # RAGシステム（知識グラフ）に因果関係を記録
                    self.rag_system.add_causal_relationship(
                        cause=previous_event,
                        effect=current_event,
                        condition=self.previous_context
                    )
                    self.just_inferred = True
                    
                    # 因果的クレジット信号の発行（強化学習エージェントへの報酬）
                    if previous_event.startswith("action_"):
                        credit_data = {
                            "type": "causal_credit",
                            "target_action": previous_event, 
                            "credit": 1.0,
                            "reason": f"Led to {current_event}"
                        }
                        print(f"  - 📢 因果的クレジット信号を生成: {credit_data}")
                        self.workspace.upload_to_workspace(
                            source="causal_engine",
                            data=credit_data,
                            salience=0.95 # 非常に高い顕著性で即座に学習させる
                        )
        
        # 状態更新
        self.previous_conscious_info = conscious_data
        self.previous_context = current_context