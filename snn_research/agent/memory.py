# /snn_research/agent/memory.py
# 日本語タイトル: エージェント・メモリ管理 (インターフェース拡張版)
# 目的: 外部スクリプトからの呼び出し整合性を確保し、エピソード記憶の記録機能を強化する。

from typing import List, Dict, Any, Optional
import logging
import os
from snn_research.cognitive_architecture.rag_snn import RAGSystem

logger = logging.getLogger(__name__)

class Memory:
    """
    エージェントの短期・長期記憶を管理するクラス。
    """
    def __init__(self, rag_system: Optional[RAGSystem] = None, memory_path: Optional[str] = None):
        # [Fix] rag_system が None の場合は新規作成し、型不一致を解消
        self.rag_system = rag_system if rag_system is not None else RAGSystem()
        self.memory_path = memory_path
        self.short_term_buffer: List[Dict[str, Any]] = []

        # 保存ディレクトリの準備
        if self.memory_path:
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)

    def record_experience(self, action: Any, result: Any, context: Optional[Dict[str, Any]] = None):
        """
        [新規追加] 経験をエピソードとして記録する。
        SelfEvolvingAgentMaster などの外部クラスから呼び出される。
        """
        episode = {
            "type": "experience",
            "action": str(action),
            "result": result,
            "context": context or {}
        }
        self.short_term_buffer.append(episode)
        logger.debug(f"Experience recorded: {action}")

        # バッファの自動固定化
        if len(self.short_term_buffer) >= 10:
            self._consolidate()

    def add_episode(self, event: str, context: Dict[str, Any]):
        """エピソードをバッファに追加する。"""
        self.short_term_buffer.append({"event": event, "context": context})
        if len(self.short_term_buffer) >= 10:
            self._consolidate()

    def _consolidate(self):
        """短期記憶を長期記憶(RAG)へ転送・固定化する。"""
        for item in self.short_term_buffer:
            text_content = f"Experience: {item.get('action', item.get('event'))}, Context: {item.get('context')}"
            
            # 文書として追加
            self.rag_system.add_knowledge(
                text=text_content, 
                metadata={"source": "memory_system", "type": "consolidated_episode"}
            )
            
            # 関係性(トリプル)として追加
            self.rag_system.add_triple(
                subj="agent",
                pred="experienced",
                obj=str(item.get('action', 'event')),
                metadata={"result": str(item.get('result', 'none'))}
            )
            
        self.short_term_buffer.clear()
        logger.info("💾 Episodic memory consolidated into RAG store.")
