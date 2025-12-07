# ファイルパス: snn_research/cognitive_architecture/causal_inference_engine.py
# Title: Causal Inference Engine (Contextual Learning) v14.0
# Description:
#   意識の時系列データを監視し、統計的共起と時間的順序から因果関係を推論する。
#   発見された因果律はGraphRAGに保存され、将来の予測モデルとして機能する。

from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
import logging

from .rag_snn import RAGSystem
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

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
        
        # 状態保持
        self.previous_conscious_info: Optional[Dict[str, Any]] = None
        self.previous_context: str = "ctx_neutral"
        
        # 統計カウンタ: (Context, Cause, Effect) -> Count
        self.causal_counter: Dict[Tuple[str, str, str], int] = defaultdict(int)
        
        # ワークスペースの購読
        self.workspace.subscribe(self.handle_conscious_broadcast)
        logger.info("🔍 Causal Inference Engine initialized & subscribed.")

    def _abstract_event(self, data: Any) -> Optional[str]:
        """
        意識に上った具体的なデータを、因果推論のための抽象的なイベントタグに変換する。
        例: {"valence": 0.9} -> "positive_emotion"
        """
        if not isinstance(data, dict):
            return "unknown_event"
            
        event_type = data.get("type")
        
        if event_type == "emotion":
            v = data.get("valence", 0)
            if v > 0.5: return "strong_positive_emotion"
            if v < -0.5: return "strong_negative_emotion"
            return "neutral_emotion"
            
        elif event_type in ["perception", "visual_perception"]:
            # 知覚対象の抽象化
            detected = data.get("detected_objects")
            if detected and isinstance(detected, list):
                labels = sorted([d.get('label', 'obj') for d in detected])
                return f"see_{'_'.join(labels[:2])}"
            return "visual_stimulus"
            
        elif event_type == "goal_setting":
            return f"goal_set_{data.get('reason', 'misc')}"
            
        # アクションの実行結果
        elif "action" in data and "status" in data:
            return f"action_{data['action']}_{data['status']}"
            
        return None

    def _get_current_context(self) -> str:
        """
        現在の脳内文脈（Context）を決定する。
        前頭前野の目標や、持続的な感情状態などが文脈を形成する。
        """
        # 前頭前野からの情報を取得（現在のゴール）
        # 注: GlobalWorkspace.get_informationはサイクル内のみ有効だが、
        # PFCの状態は持続的であるべき。ここでは簡易的に直近のブロードキャスト履歴を参照するか、
        # 本来はPFCモジュールへ直接問い合わせるのが理想。
        # 今回はGWのブロードキャスト履歴に頼らず、デフォルトを返す設計とする。
        return "default_context" 

    def handle_conscious_broadcast(self, source: str, conscious_data: Any):
        """
        意識に上った情報の連鎖を観察し、因果関係を更新する。
        """
        # クレジット信号自体は因果推論の対象外（ループ防止）
        if isinstance(conscious_data, dict) and conscious_data.get("type") == "causal_credit":
            return

        current_event = self._abstract_event(conscious_data)
        if not current_event:
            return

        # 直前のイベントが存在し、かつ文脈が安定している場合に因果を推論
        if self.previous_conscious_info and self.previous_context:
            prev_event = self._abstract_event(self.previous_conscious_info)
            
            if prev_event and prev_event != current_event:
                # 因果ペアのカウントアップ
                triple = (self.previous_context, prev_event, current_event)
                self.causal_counter[triple] += 1
                
                count = self.causal_counter[triple]
                
                # 閾値を超えたら「確信」として知識グラフに登録
                if count == self.inference_threshold:
                    self._crystallize_causality(self.previous_context, prev_event, current_event)

        # 状態更新
        self.previous_conscious_info = conscious_data
        # 文脈はイベントによって切り替わる可能性があるが、ここでは簡易的に保持
        # 本来はPFCのゴール変更イベント等で更新する
        if source == "prefrontal_cortex" and isinstance(conscious_data, dict):
            reason = conscious_data.get("reason", "misc")
            self.previous_context = f"ctx_{reason}"

    def _crystallize_causality(self, context: str, cause: str, effect: str):
        """
        十分に観測された因果関係を知識グラフに固定し、報酬信号を発行する。
        """
        logger.info(f"🔥 Causal Link Discovered: [{context}] {cause} -> {effect}")
        
        # 1. RAGへの登録
        self.rag_system.add_causal_relationship(
            cause=cause,
            effect=effect,
            condition=context
        )
        
        # 2. 因果的クレジット信号の発行
        # これにより、単なる相関が「報酬」としてエージェントの行動を強化する
        credit_data = {
            "type": "causal_credit",
            "target_action": cause, # 原因が自分の行動であれば、それを強化
            "credit": 1.0,
            "reason": f"Consistently leads to {effect} in {context}"
        }
        
        # 強い顕著性で即座に意識に上げる
        self.workspace.upload_to_workspace(
            source="causal_inference_engine",
            data=credit_data,
            salience=1.0 
        )
