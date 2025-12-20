# ファイルパス: snn_research/cognitive_architecture/causal_inference_engine.py
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
import logging

from .rag_snn import RAGSystem
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

class CausalStats:
    # (変更なし)
    def __init__(self):
        self.n11 = 0.0
        self.n10 = 0.0
        self.n01 = 0.0
        self.n00 = 0.0
        
    def update(self, cause_present: bool, effect_present: bool):
        if cause_present and effect_present:
            self.n11 += 1.0
        elif cause_present and not effect_present:
            self.n10 += 1.0
        elif not cause_present and effect_present:
            self.n01 += 1.0
        else:
            self.n00 += 1.0
            
    @property
    def delta_p(self) -> float:
        p_e_given_c = 0.0
        if (self.n11 + self.n10) > 0:
            p_e_given_c = self.n11 / (self.n11 + self.n10)
            
        p_e_given_not_c = 0.0
        if (self.n01 + self.n00) > 0:
            p_e_given_not_c = self.n01 / (self.n01 + self.n00)
            
        return p_e_given_c - p_e_given_not_c

    @property
    def total_observations(self) -> float:
        return self.n11 + self.n10 + self.n01 + self.n00


class CausalInferenceEngine:
    # [Fix 7] クラス変数型ヒント
    workspace: GlobalWorkspace

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
        
        self.previous_conscious_info: Optional[Dict[str, Any]] = None
        self.previous_context: str = "ctx_neutral"
        self.previous_event: Optional[str] = None
        
        self.causal_stats: Dict[Tuple[str, str, str], CausalStats] = defaultdict(CausalStats)
        
        self.known_events: set = set()
        
        # [Fix 7] subscribe
        self.workspace.subscribe(self.handle_conscious_broadcast)
        logger.info("🔍 Causal Inference Engine (Delta-P) initialized.")

    # (中略: メソッドは変更なし)
    def _abstract_event(self, data: Any) -> Optional[str]:
        if not isinstance(data, dict):
            return "unknown_event"
        event_type = data.get("type")
        if event_type == "emotion":
            v = data.get("valence", 0)
            if v > 0.5: return "emotion_positive"
            if v < -0.5: return "emotion_negative"
            return "emotion_neutral"
        elif event_type in ["perception", "visual_perception"]:
            detected = data.get("detected_objects")
            if detected and isinstance(detected, list):
                labels = sorted([d.get('label', 'obj') for d in detected])
                if labels:
                    return f"see_{labels[0]}"
            return "visual_stimulus"
        elif event_type == "goal_setting":
            return f"goal_{data.get('reason', 'misc')}"
        elif "action" in data and "status" in data:
            return f"action_{data['action']}_{data['status']}"
        return None

    def handle_conscious_broadcast(self, source: str, conscious_data: Any):
        if isinstance(conscious_data, dict) and conscious_data.get("type") == "causal_credit":
            return

        current_event = self._abstract_event(conscious_data)
        if not current_event:
            return
            
        self.known_events.add(current_event)

        if self.previous_event and self.previous_context:
            triple = (self.previous_context, self.previous_event, current_event)
            self.causal_stats[triple].update(cause_present=True, effect_present=True)
            
            for (ctx, cause, effect), stats in self.causal_stats.items():
                if ctx == self.previous_context and cause == self.previous_event and effect != current_event:
                    stats.update(cause_present=True, effect_present=False)

            stats = self.causal_stats[triple]
            if stats.total_observations >= self.min_observations:
                delta_p = stats.delta_p
                if delta_p > self.inference_threshold:
                    self._crystallize_causality(self.previous_context, self.previous_event, current_event, delta_p)

        self.previous_conscious_info = conscious_data
        self.previous_event = current_event
        
        if source == "prefrontal_cortex" and isinstance(conscious_data, dict):
            reason = conscious_data.get("reason", "misc")
            self.previous_context = f"ctx_{reason}"

    def _crystallize_causality(self, context: str, cause: str, effect: str, strength: float):
        logger.info(f"🔥 Causal Link Discovered (Delta-P={strength:.2f}): [{context}] {cause} -> {effect}")
        
        self.rag_system.add_causal_relationship(
            cause=cause,
            effect=effect,
            condition=f"{context} (strength={strength:.2f})"
        )
        
        credit_data = {
            "type": "causal_credit",
            "target_action": cause, 
            "credit": strength, 
            "reason": f"Strong causal link to {effect}"
        }
        
        self.workspace.upload_to_workspace(
            source="causal_inference_engine",
            data=credit_data,
            salience=1.0 
        )