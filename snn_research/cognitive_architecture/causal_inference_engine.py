# /snn_research/cognitive_architecture/causal_inference_engine.py
# 日本語タイトル: 因果推論エンジン (型整合・Delta-P強化版)
# 目的: 意識のストリームから統計的因果を抽出し、正しい型でRAGシステムへ結晶化する。

from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
import logging

from .rag_snn import RAGSystem
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

class CausalStats:
    """因果統計を保持・計算するクラス。"""
    def __init__(self):
        self.n11 = 0.0 # 原因あり、結果あり
        self.n10 = 0.0 # 原因あり、結果なし
        self.n01 = 0.0 # 原因なし、結果あり
        self.n00 = 0.0 # 原因なし、結果なし
        
    def update(self, cause_present: bool, effect_present: bool):
        if cause_present and effect_present: self.n11 += 1.0
        elif cause_present and not effect_present: self.n10 += 1.0
        elif not cause_present and effect_present: self.n01 += 1.0
        else: self.n00 += 1.0
            
    @property
    def delta_p(self) -> float:
        """Delta-P (P(E|C) - P(E|~C)) を計算する。"""
        p_e_given_c = self.n11 / (self.n11 + self.n10) if (self.n11 + self.n10) > 0 else 0.0
        p_e_given_not_c = self.n01 / (self.n01 + self.n00) if (self.n01 + self.n00) > 0 else 0.0
        return p_e_given_c - p_e_given_not_c

    @property
    def total_observations(self) -> float:
        return self.n11 + self.n10 + self.n01 + self.n00


class CausalInferenceEngine:
    """意識の遷移から動的に因果グラフを構築するエンジン。"""
    def __init__(
        self,
        rag_system: RAGSystem,
        workspace: GlobalWorkspace,
        inference_threshold: float = 0.6,
        min_observations: int = 10
    ):
        self.rag_system = rag_system
        self.workspace = workspace
        self.inference_threshold = inference_threshold
        self.min_observations = min_observations
        
        self.previous_context: str = "ctx_neutral"
        self.previous_event: Optional[str] = None
        self.causal_stats: Dict[Tuple[str, str, str], CausalStats] = defaultdict(CausalStats)
        
        # ワークスペースの放送をサブスクライブ
        self.workspace.subscribe(self.handle_conscious_broadcast)
        logger.info("🔍 Causal Inference Engine initialized.")

    def _abstract_event(self, data: Any) -> Optional[str]:
        """生データからイベントの抽象ラベルを生成。"""
        if not isinstance(data, dict): return "unknown_event"
        etype = data.get("type")
        if etype == "emotion":
            v = data.get("valence", 0)
            return "emo_pos" if v > 0.5 else "emo_neg" if v < -0.5 else "emo_neu"
        elif etype in ["perception", "visual_perception"]:
            detected = data.get("detected_objects")
            if detected and isinstance(detected, list):
                return f"see_{detected[0].get('label', 'obj')}"
            return "visual_stimulus"
        return etype

    def handle_conscious_broadcast(self, source: str, conscious_data: Any):
        """ワークスペースからの放送を処理し、因果統計を更新する。"""
        if isinstance(conscious_data, dict) and conscious_data.get("type") == "causal_credit":
            return

        current_event = self._abstract_event(conscious_data)
        if not current_event: return

        # 前のイベントとの連鎖を確認
        if self.previous_event and self.previous_context:
            triple = (self.previous_context, self.previous_event, current_event)
            self.causal_stats[triple].update(cause_present=True, effect_present=True)
            
            # 同一コンテキスト内での他のイベントとの排他性を更新
            stats = self.causal_stats[triple]
            if stats.total_observations >= self.min_observations:
                dp = stats.delta_p
                if dp > self.inference_threshold:
                    self._crystallize_causality(self.previous_context, self.previous_event, current_event, dp)

        self.previous_event = current_event
        if source == "prefrontal_cortex" and isinstance(conscious_data, dict):
            reason = conscious_data.get("reason", "misc")
            self.previous_context = f"ctx_{reason}"

    def _crystallize_causality(self, context: str, cause: str, effect: str, strength: float):
        """[mypy修正版] 発見した因果関係をRAGシステムへ永続化し、ワークスペースへ通知する。"""
        logger.info(f"🔥 Causal Link Discovered: {cause} -> {effect} (strength={strength:.2f})")
        
        # ◾️ 修正箇所: 第3引数に float 型の strength を直接渡す
        self.rag_system.add_causal_relationship(
            cause=cause,
            effect=effect,
            strength=strength
        )
        
        credit_data = {
            "type": "causal_credit",
            "cause": cause, 
            "effect": effect,
            "strength": strength,
            "context": context
        }
        
        # ワークスペースへ推論結果をフィードバック
        self.workspace.upload_to_workspace(
            source="causal_inference_engine",
            data=credit_data,
            salience=0.8
        )
