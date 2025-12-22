# ファイルパス: snn_research/cognitive_architecture/cortex.py
# Title: 大脳皮質 (インターフェース整合版)
# Description:
#   長期的な知識を管理する大脳皮質モジュール。
#   ArtificialBrainからの呼び出し(retrieve)に対応し、入力型を柔軟に扱う。

import torch
import re
from typing import Dict, Any, Optional, List, Tuple, Union
from .rag_snn import RAGSystem

class Cortex:
    """
    長期的な知識を管理する大脳皮質モジュール。
    RAGSystem (GraphRAG) をバックエンドとして使用する。
    """
    def __init__(self, rag_system: Optional[RAGSystem] = None) -> None:
        self.rag_system = rag_system
        self.knowledge_graph_legacy: Dict[str, List[Dict[str, Any]]] = {}
        print("🧠 大脳皮質（長期記憶・GraphRAG連携）モジュールが初期化されました。")

    def retrieve(self, query: Union[str, torch.Tensor]) -> List[str]:
        """
        ArtificialBrain等の外部モジュールから知識を検索するための共通インターフェース。
        
        Args:
            query (Union[str, torch.Tensor]): 検索キーワード、または特徴ベクトル。
        """
        # テンソルが渡された場合は、簡易的に最大値のインデックス等に基づいた文字列に変換
        # (将来的にベクトル検索を実装するためのプレースホルダ)
        if isinstance(query, torch.Tensor):
            if query.ndim > 0:
                concept_id = int(torch.argmax(query).item())
                search_term = f"concept_{concept_id}"
            else:
                search_term = "unknown_signal"
        else:
            search_term = str(query)

        return self.retrieve_knowledge(search_term)

    def retrieve_knowledge(self, concept: str) -> List[str]:
        """概念に関連する知識を検索する"""
        if self.rag_system:
            try:
                return self.rag_system.search(concept)
            except Exception as e:
                print(f"⚠️ Cortex search error: {e}")
                return []
        return []

    def consolidate_memory(self, episode: Dict[str, Any]) -> None:
        """
        短期記憶のエピソードを解釈し、ナレッジグラフに統合する。
        """
        source_input = episode.get("source_input")
        content = episode.get("content")
        
        if episode.get('type') == 'knowledge_correction':
            concept = episode.get('concept')
            new_info = content.get('text') if isinstance(content, dict) else str(content)
            if concept and new_info and self.rag_system:
                self.rag_system.update_knowledge(concept, "is_defined_as", new_info, reason=episode.get('reason', 'correction'))
            return

        text_to_process = ""
        if isinstance(source_input, str):
            text_to_process = source_input
        elif isinstance(content, str):
            text_to_process = content
            
        if not text_to_process:
            return

        print(f"🤔 大脳皮質: エピソード '{text_to_process[:30]}...' から知識構造化を試みています...")

        triples = self._extract_triples(text_to_process)
        
        if triples and self.rag_system:
            for subj, rel, obj in triples:
                self.rag_system.add_triple(subj, rel, obj, metadata={"source": "consolidation"})
        else:
            keywords = set(re.findall(r'\b[a-zA-Z]{5,}\b', text_to_process.lower()))
            if len(keywords) > 1:
                k_list = list(keywords)
                for i in range(len(k_list)-1):
                    self._add_relationship_legacy(k_list[i], "co-occurred_with", k_list[i+1])

    def _extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """簡易ルールベースによる抽出 (日/英対応)"""
        triples = []
        text = text.strip()
        
        patterns_en = [
            (r'(.+?)\s+is\s+a\s+(.+)', 'is_a'),
            (r'(.+?)\s+is\s+(.+)', 'is'),
            (r'(.+?)\s+has\s+(.+)', 'has'),
            (r'(.+?)\s+causes\s+(.+)', 'causes'),
            (r'(.+?)\s+likes\s+(.+)', 'likes'),
        ]

        patterns_ja = [
            (r'(.+?)は(.+?)の一種です', 'is_a'),
            (r'(.+?)は(.+?)です', 'is'),
            (r'(.+?)が(.+?)にある', 'located_in'),
            (r'(.+?)が(.+?)する', 'does'),
            (r'(.+?)は(.+?)を持つ', 'has'),
            (r'(.+?)は(.+?)が好き', 'likes'),
        ]
        
        for pattern, relation in patterns_en:
            clean_text = text.rstrip('.!?')
            match = re.search(pattern, clean_text, re.IGNORECASE)
            if match:
                subj = match.group(1).strip()
                obj = match.group(2).strip()
                if len(subj.split()) <= 5 and len(obj.split()) <= 10:
                     triples.append((subj, relation, obj))
        
        for pattern, relation in patterns_ja:
            clean_text = text.rstrip('。！？')
            match = re.search(pattern, clean_text)
            if match:
                subj = match.group(1).strip()
                obj = match.group(2).strip()
                if len(subj) <= 20 and len(obj) <= 30:
                     triples.append((subj, relation, obj))
        
        return triples

    def _add_relationship_legacy(self, source: str, relation: str, target: Any) -> None:
        if source not in self.knowledge_graph_legacy:
            self.knowledge_graph_legacy[source] = []
        if not any(r['relation'] == relation and r['target'] == target for r in self.knowledge_graph_legacy[source]):
            self.knowledge_graph_legacy[source].append({"relation": relation, "target": target})

    def get_all_knowledge(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.knowledge_graph_legacy
