# ファイルパス: snn_research/cognitive_architecture/symbol_grounding.py
# Title: Deep Symbol Grounding (Bi-directional & ART-based) v14.1
# Description:
#   ニューラル活動パターン（アトラクタ）と知識グラフ上の概念（シンボル）を
#   動的かつ双方向にリンクさせる。
#   修正点:
#   - recall_pattern: シンボルからニューラルパターンを復元するトップダウン機能を追加。
#   - ART (Adaptive Resonance Theory) に基づく vigilance パラメータを導入し、
#     類似度判定の厳しさを動的に調整可能にした。

import hashlib
import torch
from typing import Dict, Any, Optional
import logging
from .rag_snn import RAGSystem

logger = logging.getLogger(__name__)

class SymbolGrounding:
    """
    ニューラルパターン(Sub-symbolic)と知識グラフ(Symbolic)の相互変換を行う。
    Adaptive Resonance Theory (ART) の警戒パラメータ(vigilance)を用いて
    カテゴリ形成（記号創発）の粒度を制御する。
    """
    def __init__(self, rag_system: RAGSystem, base_vigilance: float = 0.85):
        self.rag_system = rag_system
        self.base_vigilance = base_vigilance
        self.current_vigilance = base_vigilance
        
        # ConceptID -> Pattern Centroid (Tensor) のマッピング
        self.known_patterns: Dict[str, torch.Tensor] = {} 
        self.concept_counter = 0
        
        logger.info(f"⚓ SymbolGrounding initialized (Vigilance: {base_vigilance}).")

    def set_vigilance(self, vigilance: float):
        """
        警戒パラメータを設定する。
        高い値(>0.9): 細かい違いを区別し、新しい概念を作りやすくなる。
        低い値(<0.7): 大雑把に分類し、既存概念に当てはめやすくなる。
        """
        self.current_vigilance = max(0.0, min(1.0, vigilance))
        # logger.debug(f"   ⚓ Vigilance updated to {self.current_vigilance:.2f}")

    def _find_nearest_concept(self, pattern: torch.Tensor) -> Optional[str]:
        """
        既存の概念パターンの中で、現在のパターンに類似しているものを検索する。
        """
        if not self.known_patterns:
            return None
            
        pattern_flat = pattern.view(-1).float()
        best_concept = None
        max_sim = -1.0
        
        for concept_id, centroid in self.known_patterns.items():
            centroid_flat = centroid.view(-1).float().to(pattern.device)
            # 次元が違う場合はスキップ（またはリサイズ）
            if centroid_flat.shape != pattern_flat.shape:
                continue
            
            # コサイン類似度計算
            sim = torch.nn.functional.cosine_similarity(pattern_flat.unsqueeze(0), centroid_flat.unsqueeze(0)).item()
            if sim > max_sim:
                max_sim = sim
                best_concept = concept_id
                
        # 現在の vigilance よりも類似度が高ければマッチとみなす
        if max_sim >= self.current_vigilance:
            return best_concept
        return None

    def ground_neural_pattern(self, pattern: torch.Tensor, context: str) -> str:
        """
        [Bottom-up] ニューラル活動パターンをシンボルとして接地する。
        ARTロジックに基づき、既存概念への同化(Assimilation)か、
        新規概念の調節(Accommodation/Emergence)かを選択する。
        """
        # 1. 既存概念との照合
        existing_concept = self._find_nearest_concept(pattern)
        
        if existing_concept:
            # 既存概念の更新 (Online Average: パターンの重心を移動)
            # 学習率 alpha は固定または適応的に
            alpha = 0.1
            old_centroid = self.known_patterns[existing_concept].to(pattern.device)
            self.known_patterns[existing_concept] = (1 - alpha) * old_centroid + alpha * pattern
            
            # コンテキストの強化
            self.rag_system.add_triple(existing_concept, "re-observed_in", context)
            return existing_concept
        
        # 2. 新規概念の創発 (Emergence)
        self.concept_counter += 1
        new_concept_id = f"neural_concept_{self.concept_counter:04d}"
        
        # パターンの登録
        self.known_patterns[new_concept_id] = pattern.detach().clone()
        
        logger.info(f"✨ 記号創発 (Emergence): ニューラル活動 -> 新シンボル '{new_concept_id}' (Ctx: {context})")

        # 3. グラフへの構造化登録
        self.rag_system.add_triple(new_concept_id, "is_a", "neural_attractor")
        self.rag_system.add_triple(new_concept_id, "emerged_in_context", context)
        
        # 統計情報の記録
        activation_level = pattern.float().mean().item()
        self.rag_system.add_triple(new_concept_id, "has_activation_level", f"{activation_level:.3f}")
        
        return new_concept_id

    def recall_pattern(self, concept_id: str) -> Optional[torch.Tensor]:
        """
        [Top-down] 概念IDから、それに対応するニューラル活動パターン（プロトタイプ）を取り出す。
        想像(Imagination)やプライミング(Priming)に使用される。
        """
        if concept_id in self.known_patterns:
            return self.known_patterns[concept_id]
        
        # IDが直接見つからない場合、RAGを使って類似概念や定義を検索し、
        # それに関連するパターンを合成する高度なロジックも考えられるが、
        # ここでは簡易的にNoneを返す。
        return None

    def process_observation(self, observation: Dict[str, Any], context: str) -> str:
        """
        [Bottom-up] 外部観測データ（辞書など）をシンボル化する。
        """
        obs_str = str(sorted(observation.items()))
        obs_hash = hashlib.md5(obs_str.encode()).hexdigest()
        concept_id = f"observation_{obs_hash[:8]}"
        
        self.rag_system.add_triple(concept_id, "is_a", "observation_event")
        self.rag_system.add_triple(concept_id, "observed_in", context)
        
        for key, val in observation.items():
            if isinstance(val, (str, int, float, bool)):
                self.rag_system.add_triple(concept_id, f"has_{key}", str(val))
                
        return concept_id