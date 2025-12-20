# ファイルパス: snn_research/agent/memory.py
# Title: Agent Memory System v1.1
# Description: RAGSystem APIとの整合性を修正。

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import time
import json
import logging

if TYPE_CHECKING:
    from snn_research.cognitive_architecture.rag_snn import RAGSystem

logger = logging.getLogger(__name__)

class Memory:
    """
    エージェントの記憶システム。
    短期記憶（リスト）と長期記憶（RAG/VectorDB）を統合管理する。
    """
    def __init__(self, rag_system: Optional['RAGSystem'] = None, capacity: int = 100, memory_path: str = "runs/agent_memory.jsonl"):
        self.short_term_memory: List[Dict[str, Any]] = []
        self.capacity = capacity
        self.rag_system = rag_system
        self.long_term_storage: List[Dict[str, Any]] = [] 
        self.memory_path = memory_path

        # 既存の記憶をロード
        self._load_memory()

    def _load_memory(self):
        # 簡易実装: ファイルがあれば読み込む
        try:
            with open(self.memory_path, 'r') as f:
                for line in f:
                    self.long_term_storage.append(json.loads(line))
        except FileNotFoundError:
            pass

    def add(self, entry: Dict[str, Any]):
        """短期記憶に追加"""
        entry['timestamp'] = time.time()
        self.short_term_memory.append(entry)
        if len(self.short_term_memory) > self.capacity:
            oldest = self.short_term_memory.pop(0)
            self._consolidate(oldest)

    def record_experience(
        self,
        state: Dict[str, Any],
        action: str,
        result: Any,
        reward: Dict[str, Any],
        expert_used: List[str],
        decision_context: Dict[str, Any],
        causal_snapshot: Optional[str] = None
    ):
        """経験を詳細に記録する（互換性維持）"""
        entry = {
            "state": state,
            "action": action,
            "result": str(result),
            "reward": reward,
            "context": decision_context,
            "causal": causal_snapshot
        }
        self.add(entry)
        
        # ファイルにも追記
        try:
            import os
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
            with open(self.memory_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _consolidate(self, entry: Dict[str, Any]):
        """長期記憶（RAG）へ転送"""
        if self.rag_system:
            text_content = json.dumps(entry, ensure_ascii=False)
            # 修正: add_knowledge -> add_document
            self.rag_system.add_document(text_content, metadata={"source": "memory_consolidation"})
            
            # 関係性としても追加
            if "action" in entry and "result" in entry:
                self.rag_system.add_relationship(
                    source=f"Action:{entry['action']}",
                    relation="resulted_in",
                    target=f"Result:{str(entry['result'])[:50]}"
                )
        else:
            self.long_term_storage.append(entry)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """検索"""
        results = []
        # 短期記憶から
        for item in reversed(self.short_term_memory):
            if query in str(item):
                results.append(str(item))
                if len(results) >= top_k: return results
        
        # 長期記憶（RAG）から
        if self.rag_system:
            # 修正: query -> search
            rag_results = self.rag_system.search(query, k=top_k - len(results))
            results.extend(rag_results)
            
        return results
    
    def retrieve_successful_experiences(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """成功体験（報酬が高いもの）を取得"""
        # ファイルから読み込んでソート
        candidates = self.long_term_storage + self.short_term_memory
        
        def get_reward(item):
            r = item.get('reward', 0)
            if isinstance(r, dict): return float(r.get('external', 0))
            return float(r)

        candidates.sort(key=get_reward, reverse=True)
        return candidates[:top_k]