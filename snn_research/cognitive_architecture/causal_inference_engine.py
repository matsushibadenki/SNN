# ファイルパス: snn_research/cognitive_architecture/causal_inference_engine.py
# Title: Causal Inference Engine (Delta-P Rule) v14.1
# Description:
#   意識の時系列データを監視し、統計的因果推論（Delta-P）を用いて強固な因果関係を発見する。
#   修正点:
#   - 単純なカウンターではなく、分割表（Contingency Table）を用いた統計的判定を導入。
#   - P(E|C) - P(E|~C) を計算し、因果力 (Causal Power) を評価する。

from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
import logging

from .rag_snn import RAGSystem
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

class CausalStats:
    """因果関係の統計情報を保持するクラス"""
    def __init__(self):
        # N(C, E): 原因あり、結果あり
        self.n11 = 0.0
        # N(C, ~E): 原因あり、結果なし
        self.n10 = 0.0
        # N(~C, E): 原因なし、結果あり
        self.n01 = 0.0
        # N(~C, ~E): 原因なし、結果なし（背景）
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
        """
        Delta-P 指標を計算する。
        Delta-P = P(E|C) - P(E|~C)
        1.0に近いほど強い正の因果、-1.0に近いほど強い抑制的因果、0は無関係。
        """
        # P(E|C) = N11 / (N11 + N10)
        p_e_given_c = 0.0
        if (self.n11 + self.n10) > 0:
            p_e_given_c = self.n11 / (self.n11 + self.n10)
            
        # P(E|~C) = N01 / (N01 + N00)
        p_e_given_not_c = 0.0
        if (self.n01 + self.n00) > 0:
            p_e_given_not_c = self.n01 / (self.n01 + self.n00)
            
        return p_e_given_c - p_e_given_not_c

    @property
    def total_observations(self) -> float:
        return self.n11 + self.n10 + self.n01 + self.n00


class CausalInferenceEngine:
    """
    意識の連鎖を観察し、文脈依存の因果関係を推論して知識グラフを構築するエンジン。
    """
    def __init__(
        self,
        rag_system: RAGSystem,
        workspace: GlobalWorkspace,
        inference_threshold: float = 0.6, # Delta-Pの閾値
        min_observations: int = 10        # 最低観測回数
    ):
        self.rag_system = rag_system
        self.workspace = workspace
        self.inference_threshold = inference_threshold
        self.min_observations = min_observations
        
        # 状態保持
        self.previous_conscious_info: Optional[Dict[str, Any]] = None
        self.previous_context: str = "ctx_neutral"
        self.previous_event: Optional[str] = None
        
        # 統計データ: (Context, Cause, Effect) -> CausalStats
        # ここでは簡易化のため、特定のCauseに対して全Effect候補の統計をとる構造にするのが理想だが
        # メモリ効率のため、観測されたペアに対して統計を維持する
        self.causal_stats: Dict[Tuple[str, str, str], CausalStats] = defaultdict(CausalStats)
        
        # 既知のイベントセット（~Cの判定に使用）
        self.known_events: set = set()
        
        # ワークスペースの購読
        self.workspace.subscribe(self.handle_conscious_broadcast)
        logger.info("🔍 Causal Inference Engine (Delta-P) initialized.")

    def _abstract_event(self, data: Any) -> Optional[str]:
        """
        意識に上った具体的なデータを、因果推論のための抽象的なイベントタグに変換する。
        """
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
                # 主要な物体のみをイベントとする
                if labels:
                    return f"see_{labels[0]}"
            return "visual_stimulus"
            
        elif event_type == "goal_setting":
            return f"goal_{data.get('reason', 'misc')}"
            
        elif "action" in data and "status" in data:
            return f"action_{data['action']}_{data['status']}"
            
        return None

    def handle_conscious_broadcast(self, source: str, conscious_data: Any):
        """
        意識に上った情報の連鎖を観察し、因果関係統計を更新する。
        """
        # クレジット信号自体は因果推論の対象外
        if isinstance(conscious_data, dict) and conscious_data.get("type") == "causal_credit":
            return

        current_event = self._abstract_event(conscious_data)
        if not current_event:
            return
            
        self.known_events.add(current_event)

        # 直前のイベントが存在し、かつ文脈が安定している場合に統計を更新
        if self.previous_event and self.previous_context:
            
            # --- 1. ポジティブケース (C -> E) の更新 ---
            # 前回のイベント(C) -> 今回のイベント(E)
            triple = (self.previous_context, self.previous_event, current_event)
            self.causal_stats[triple].update(cause_present=True, effect_present=True)
            
            # --- 2. ネガティブケース (C -> ~E) や (~C -> E) の更新 ---
            # 統計的信頼性を高めるには「起きなかったこと」もカウントする必要がある。
            # 簡易実装として、現在着目している主要な因果候補についてのみ更新を行う。
            # (すべての組み合わせを更新すると計算量が爆発するため)
            
            # このコンテキストで過去に観測された他の結果 E' について、
            # Cが起きたのにE'が起きなかった (C -> ~E') ケースとしてカウント
            for (ctx, cause, effect), stats in self.causal_stats.items():
                if ctx == self.previous_context and cause == self.previous_event and effect != current_event:
                    stats.update(cause_present=True, effect_present=False)

            # --- 3. 因果の結晶化判定 ---
            stats = self.causal_stats[triple]
            if stats.total_observations >= self.min_observations:
                delta_p = stats.delta_p
                # 強い因果関係があれば登録
                if delta_p > self.inference_threshold:
                    self._crystallize_causality(self.previous_context, self.previous_event, current_event, delta_p)

        # 状態更新
        self.previous_conscious_info = conscious_data
        self.previous_event = current_event
        
        # 文脈の更新 (PFCからのゴール設定など)
        if source == "prefrontal_cortex" and isinstance(conscious_data, dict):
            reason = conscious_data.get("reason", "misc")
            self.previous_context = f"ctx_{reason}"

    def _crystallize_causality(self, context: str, cause: str, effect: str, strength: float):
        """
        十分に観測された因果関係を知識グラフに固定し、報酬信号を発行する。
        """
        logger.info(f"🔥 Causal Link Discovered (Delta-P={strength:.2f}): [{context}] {cause} -> {effect}")
        
        # 1. RAGへの登録
        self.rag_system.add_causal_relationship(
            cause=cause,
            effect=effect,
            condition=f"{context} (strength={strength:.2f})"
        )
        
        # 2. 因果的クレジット信号の発行
        # 原因が自分の行動であれば、それを強化
        credit_data = {
            "type": "causal_credit",
            "target_action": cause, 
            "credit": strength, # 因果の強さを報酬量とする
            "reason": f"Strong causal link to {effect}"
        }
        
        # 強い顕著性で即座に意識に上げる
        self.workspace.upload_to_workspace(
            source="causal_inference_engine",
            data=credit_data,
            salience=1.0 
        )
