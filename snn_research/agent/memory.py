# /snn_research/agent/memory.py
# 日本語タイトル: エージェント・メモリ管理 (RAG連携修正版)
# 目的: エピソード記憶を長期記憶(RAG)へ転送する際のインターフェース不整合を解消。

from typing import List, Dict, Any, Optional
import logging
from snn_research.cognitive_architecture.rag_snn import RAGSystem

logger = logging.getLogger(__name__)

class Memory:
    """
    エージェントの短期・長期記憶を管理するクラス。
    """
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.short_term_buffer: List[Dict[str, Any]] = []

    def add_episode(self, event: str, context: Dict[str, Any]):
        """短期記憶バッファにエピソードを追加。"""
        self.short_term_buffer.append({"event": event, "context": context})
        
        # バッファが一定量を超えたら固定化を実行
        if len(self.short_term_buffer) > 10:
            self._consolidate()

    def _consolidate(self):
        """
        [mypy修正] 短期記憶から長期記憶(RAG)への転送。
        存在しない add_document / add_relationship を add_knowledge / add_triple に修正。
        """
        for item in self.short_term_buffer:
            text_content = f"Event: {item['event']}, Context: {item['context']}"
            
            # 1. 文書として追加 (add_document の代わりに add_knowledge を使用)
            self.rag_system.add_knowledge(
                text=text_content, 
                metadata={"source": "memory_consolidation", "type": "episode"}
            )
            
            # 2. 関係性として追加 (add_relationship の代わりに add_triple を使用)
            # source, target などの情報を抽出してトリプル化
            source = item['context'].get('subject', 'unknown_agent')
            relation = "executed"
            target = item['event']
            
            self.rag_system.add_triple(
                subj=source,
                pred=relation,
                obj=target,
                metadata={"type": "behavioral_history"}
            )
            
        self.short_term_buffer.clear()
        logger.info("💾 Episodic memory consolidated into RAG store.")
