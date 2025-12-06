# ファイルパス: snn_research/cognitive_architecture/symbol_grounding.py
# Title: 深層記号接地システム (Deep Symbol Grounding)
# Description:
#   ロードマップ Phase 5 実装。
#   ニューラル活動パターン（アトラクタ）と、知識グラフ上の概念（シンボル）を
#   動的かつ双方向にリンクさせる。
#   - Bottom-up: ニューロン発火 -> シンボル創発 (ハッシュ化 & クラスタリング)
#   - Top-down: シンボル -> ニューロン活性化 (プライミング)

import hashlib
import torch
import numpy as np
from typing import Set, Dict, Any, Optional, List
import logging

from .rag_snn import RAGSystem

logger = logging.getLogger(__name__)

class SymbolGrounding:
    """
    ニューラルパターン(Sub-symbolic)と知識グラフ(Symbolic)の相互変換を行う。
    """
    def __init__(self, rag_system: RAGSystem, similarity_threshold: float = 0.85):
        self.rag_system = rag_system
        self.similarity_threshold = similarity_threshold
        self.known_patterns: Dict[str, torch.Tensor] = {} # Hash -> Pattern Centroid
        self.concept_counter = 0
        
        logger.info("⚓ SymbolGrounding initialized. Bridging the gap between neurons and symbols.")

    def _get_pattern_hash(self, pattern: torch.Tensor) -> str:
        """
        スパイクパターンのハッシュを生成する（簡易的なLocality Sensitive Hashingの代用）。
        パターンを2値化してハッシュ化する。
        """
        # テンソルをCPUへ移動し、2値化
        binary_pattern = (pattern > 0.5).byte().cpu().numpy()
        pattern_bytes = binary_pattern.tobytes()
        return hashlib.sha256(pattern_bytes).hexdigest()

    def _find_nearest_concept(self, pattern: torch.Tensor) -> Optional[str]:
        """
        既存の概念パターンの中で、現在のパターンに類似しているものを検索する。
        (コサイン類似度を使用)
        """
        if not self.known_patterns:
            return None
            
        pattern_flat = pattern.view(-1).float()
        best_concept = None
        max_sim = -1.0
        
        for concept_id, centroid in self.known_patterns.items():
            centroid_flat = centroid.view(-1).float().to(pattern.device)
            if centroid_flat.shape != pattern_flat.shape:
                continue
                
            sim = torch.nn.functional.cosine_similarity(pattern_flat.unsqueeze(0), centroid_flat.unsqueeze(0)).item()
            if sim > max_sim:
                max_sim = sim
                best_concept = concept_id
                
        if max_sim >= self.similarity_threshold:
            return best_concept
        return None

    def ground_neural_pattern(self, pattern: torch.Tensor, context: str) -> str:
        """
        [Bottom-up] ニューラル活動パターンをシンボルとして接地する。
        既存の概念に近ければそれを返し、新しければ新規概念を創発させる。
        
        Args:
            pattern (torch.Tensor): 知覚野や思考層の活動テンソル。
            context (str): 発生した文脈（メタデータ）。
            
        Returns:
            str: 接地された概念ID (Concept ID)。
        """
        # 1. 既存概念との照合
        existing_concept = self._find_nearest_concept(pattern)
        
        if existing_concept:
            # 既存概念のセントロイドを更新 (オンライン学習)
            # old_centroid = self.known_patterns[existing_concept]
            # self.known_patterns[existing_concept] = 0.9 * old_centroid + 0.1 * pattern
            # logger.debug(f"  -> Grounded to existing concept: {existing_concept}")
            
            # コンテキストの強化
            self.rag_system.add_triple(existing_concept, "re-observed_in", context)
            return existing_concept
        
        # 2. 新規概念の創発
        self.concept_counter += 1
        new_concept_id = f"neural_concept_{self.concept_counter:04d}"
        
        # パターンの登録
        self.known_patterns[new_concept_id] = pattern.detach().clone()
        
        logger.info(f"✨ 記号創発 (Emergence): ニューラル活動 -> 新シンボル '{new_concept_id}'")

        # 3. グラフへの構造化登録
        self.rag_system.add_triple(new_concept_id, "is_a", "neural_attractor")
        self.rag_system.add_triple(new_concept_id, "emerged_in_context", context)
        
        # 統計情報の記録
        activation_level = pattern.float().mean().item()
        self.rag_system.add_triple(new_concept_id, "has_activation_level", f"{activation_level:.3f}")
        
        return new_concept_id

    def process_observation(self, observation: Dict[str, Any], context: str) -> str:
        """
        [Bottom-up] 外部観測データ（辞書など）をシンボル化する。
        """
        # 観測データのハッシュ化（簡易版）
        obs_str = str(sorted(observation.items()))
        obs_hash = hashlib.md5(obs_str.encode()).hexdigest()
        concept_id = f"observation_{obs_hash[:8]}"
        
        # グラフに存在しなければ登録
        # (RAGSystem側で重複チェックはあるが、ここではID生成が主)
        self.rag_system.add_triple(concept_id, "is_a", "observation_event")
        self.rag_system.add_triple(concept_id, "observed_in", context)
        
        for key, val in observation.items():
            if isinstance(val, (str, int, float, bool)):
                self.rag_system.add_triple(concept_id, f"has_{key}", str(val))
                
        return concept_id

    def get_priming_signal(self, concept_id: str) -> Optional[torch.Tensor]:
        """
        [Top-down] 概念IDから、それに対応するニューラル活動パターン（プライミング信号）を取り出す。
        「赤い」と言われて視覚野が発火するような現象を再現。
        """
        if concept_id in self.known_patterns:
            return self.known_patterns[concept_id]
        
        # RAGから関連概念を検索して、既知のパターンを探す（連想）
        related_concepts = self.rag_system.search(concept_id, k=3)
        for related in related_concepts:
            # 検索結果テキストからIDを抽出するロジックが必要だが、
            # ここでは簡易的にRAGの検索結果がIDを含んでいると仮定
            pass
            
        return None
