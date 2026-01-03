# /snn_research/agent/memory.py
# 日本語タイトル: エージェント・メモリ管理 (型安全性強化版)
# 目的: 報酬値の計算における型不一致エラーを解消し、堅牢な経験検索を実現する。

from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union
import time
import json
import logging
import os

if TYPE_CHECKING:
    from snn_research.cognitive_architecture.rag_snn import RAGSystem

logger = logging.getLogger(__name__)

class Memory:
    """
    エージェントの記憶システム。
    短期記憶（リスト）と長期記憶（RAG）を統合管理し、ファイルへの永続化もサポートする。
    """
    def __init__(self, rag_system: Optional['RAGSystem'] = None, capacity: int = 100, memory_path: str = "workspace/runs/agent_memory.jsonl"):
        self.short_term_memory: List[Dict[str, Any]] = []
        self.capacity = capacity
        self.rag_system = rag_system
        self.long_term_storage: List[Dict[str, Any]] = [] 
        self.memory_path = memory_path

        # 保存先ディレクトリの作成
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        # 既存の記憶をロード
        self._load_memory()

    def _load_memory(self):
        """ファイルから長期記憶をロードする。"""
        try:
            if os.path.exists(self.memory_path):
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.long_term_storage.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to load memory from {self.memory_path}: {e}")

    def add(self, entry: Dict[str, Any]):
        """短期記憶に追加し、キャパシティを超えたら固定化する。"""
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
        reward: Union[Dict[str, Any], float, int],
        expert_used: List[str],
        decision_context: Dict[str, Any],
        causal_snapshot: Optional[str] = None
    ):
        """経験を詳細に記録する。"""
        entry = {
            "state": state,
            "action": action,
            "result": str(result),
            "reward": reward,
            "expert_used": expert_used,
            "context": decision_context,
            "causal": causal_snapshot
        }
        self.add(entry)
        
        # ファイルへの追記（永続化）
        try:
            with open(self.memory_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write experience to file: {e}")

    def _consolidate(self, entry: Dict[str, Any]):
        """短期記憶の項目を RAGSystem (長期記憶) へ転送する。"""
        if self.rag_system:
            text_content = json.dumps(entry, ensure_ascii=False)
            
            # RAGSystem.add_knowledge を使用
            self.rag_system.add_knowledge(
                text=text_content, 
                metadata={"source": "memory_consolidation", "action": entry.get("action", "unknown")}
            )
            
            # RAGSystem.add_triple を使用して関係性を保存
            if "action" in entry and "result" in entry:
                self.rag_system.add_triple(
                    subj=f"Action:{entry['action']}",
                    pred="resulted_in",
                    obj=f"Result:{str(entry['result'])[:50]}",
                    metadata={"reward": str(entry.get("reward"))}
                )
        else:
            self.long_term_storage.append(entry)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """短期および長期記憶から検索を行う。"""
        results: List[str] = []
        for item in reversed(self.short_term_memory):
            if query in str(item):
                results.append(json.dumps(item, ensure_ascii=False))
                if len(results) >= top_k:
                    return results
        
        if self.rag_system:
            rag_results = self.rag_system.search(query, k=top_k - len(results))
            results.extend(rag_results)
            
        return results
    
    def retrieve_successful_experiences(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        [Fix] 報酬が高い成功体験を優先的に取得する。
        mypy [arg-type] エラーを解消するため、数値抽出ロジックを堅牢化。
        """
        candidates = self.long_term_storage + self.short_term_memory
        
        def get_reward_value(item: Dict[str, Any]) -> float:
            r = item.get('reward', 0.0)
            
            # 報酬が辞書形式の場合（外部報酬と内部報酬がある場合）
            if isinstance(r, dict):
                # 明示的に型チェックを行い、Noneを排除して float に変換
                ext_r = r.get('external')
                int_r = r.get('internal')
                
                if ext_r is not None:
                    return float(ext_r)
                if int_r is not None:
                    return float(int_r)
                return 0.0
            
            # 報酬が数値または文字列の場合
            try:
                return float(r) if r is not None else 0.0
            except (ValueError, TypeError):
                return 0.0

        # 計算された数値に基づいてソート
        candidates.sort(key=get_reward_value, reverse=True)
        return candidates[:top_k]