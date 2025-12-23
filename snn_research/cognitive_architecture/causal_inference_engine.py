# /snn_research/cognitive_architecture/causal_inference_engine.py
# 日本語タイトル: 因果推論エンジン (整合性修正版)
# 目的: 意識のストリームからDelta-P統計を用いて因果関係を抽出・結晶化する。

from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
import logging

from .rag_snn import RAGSystem
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

class CausalStats:
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
        p1 = self.n11 / (self.n11 + self.n10) if (self.n11 + self.n10) > 0 else 0.0
        p0 = self.n01 / (self.n01 + self.n00) if (self.n01 + self.n00) > 0 else 0.0
        return p1 - p0

    @property
    def total_observations(self) -> float:
        return self.n11 + self.n10 + self.n01 + self.n00

class CausalInferenceEngine:
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
        
        # 注意の放送を監視
        self.workspace.subscribe(self.handle_conscious_broadcast)
        logger.info("🔍 Causal Inference Engine initialized.")

    def _abstract_event(self, data: Any) -> Optional[str]:
        if not isinstance(data, dict): return "unknown_event"
        etype = data.get("type")
        if etype == "emotion":
            v = data.get("valence", 0)
            return "emo_pos" if v > 0.5 else "emo_neg" if v < -0.5 else "emo_neu"
        elif etype in ["perception", "visual_perception"]:
            detected = data.get("detected_objects")
            return f"see_{detected[0]['label']}" if detected else "visual_stimulus"
        return etype

    def handle_conscious_broadcast(self, source: str, conscious_data: Any):
        if isinstance(conscious_data, dict) and conscious_data.get("type") == "causal_credit":
            return

        current_event = self._abstract_event(conscious_data)
        if not current_event: return

        if self.previous_event and self.previous_context:
            triple = (self.previous_context, self.previous_event, current_event)
            self.causal_stats[triple].update(cause_present=True, effect_present=True)
            
            # 統計更新
            stats = self.causal_stats[triple]
            if stats.total_observations >= self.min_observations:
                dp = stats.delta_p
                if dp > self.inference_threshold:
                    self._crystallize_causality(self.previous_context, self.previous_event, current_event, dp)

        self.previous_event = current_event
        if source == "prefrontal_cortex" and isinstance(conscious_data, dict):
            self.previous_context = f"ctx_{conscious_data.get('reason', 'misc')}"

    def _crystallize_causality(self, context: str, cause: str, effect: str, strength: float):
        logger.info(f"🔥 Causal Link: {cause} -> {effect} (dp={strength:.2f})")
        self.rag_system.add_causal_relationship(cause, effect, f"{context} ({strength:.2f})")
        
        # ワークスペースへ結果をアップロード (mypyエラー箇所修正)
        self.workspace.upload_to_workspace(
            source="causal_inference_engine",
            data={"type": "causal_credit", "cause": cause, "effect": effect, "strength": strength},
            salience=1.0
        )
