# ファイルパス: snn_research/cognitive_architecture/symbol_grounding.py
# (更新: グラフ構造への接地)
# Title: 記号創発システム (Graph Grounding)
# Description:
# - ニューラル活動や観測データから創発されたシンボルを、
#   ナレッジグラフ上のノードとして定義し、他の概念と接続する。

from typing import Set, Dict, Any
import hashlib
import torch

from .rag_snn import RAGSystem

class SymbolGrounding:
    """
    観測から新しいシンボルを創発し、ナレッジグラフに定着させるシステム。
    """
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.known_concepts: Set[str] = set()
        self.concept_counter = 100

    def _get_pattern_hash(self, pattern: torch.Tensor) -> str:
        pattern_bytes = pattern.cpu().numpy().tobytes()
        return hashlib.sha256(pattern_bytes).hexdigest()

    def _get_observation_hash(self, observation: Dict[str, Any]) -> str:
        s = str(sorted(observation.items()))
        return hashlib.sha256(s.encode()).hexdigest()

    def ground_neural_pattern(self, pattern: torch.Tensor, context: str):
        """
        新しいニューラル活動パターンをグラフ上のシンボルとして接地する。
        """
        pattern_hash = self._get_pattern_hash(pattern)

        if pattern_hash not in self.known_concepts:
            self.known_concepts.add(pattern_hash)
            new_concept_id = f"neural_concept_{self.concept_counter}"
            self.concept_counter += 1

            print(f"✨ 記号創発: ニューラル活動 -> シンボル '{new_concept_id}'")

            # グラフへの構造化登録
            # ノード: new_concept_id
            # エッジ: is_a -> neural_pattern
            self.rag_system.add_triple(new_concept_id, "is_a", "neural_pattern")
            
            # エッジ: observed_in -> context
            self.rag_system.add_triple(new_concept_id, "observed_in", context)
            
            # 属性情報 (平均発火率など) も関係性として記録
            activation_level = f"level_{pattern.float().mean().item():.2f}"
            self.rag_system.add_triple(new_concept_id, "has_activation", activation_level)

    def process_observation(self, observation: Dict[str, Any], context: str):
        """
        外部観測をグラフ上のシンボルとして接地する。
        """
        if not isinstance(observation, dict):
            return

        obs_hash = self._get_observation_hash(observation)

        if obs_hash not in self.known_concepts:
            self.known_concepts.add(obs_hash)
            new_concept_id = f"observation_{self.concept_counter}"
            self.concept_counter += 1
            
            print(f"✨ 記号創発: 観測データ -> シンボル '{new_concept_id}'")

            self.rag_system.add_triple(new_concept_id, "is_a", "observation")
            self.rag_system.add_triple(new_concept_id, "occurred_in", context)

            # 観測内容の詳細をグラフに展開
            for key, value in observation.items():
                if isinstance(value, (str, int, float)):
                    # "observation_101" --[has_color]--> "red"
                    self.rag_system.add_triple(new_concept_id, f"has_{key}", str(value))
