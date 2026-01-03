# /snn_research/cognitive_architecture/cortex.py
# 日本語タイトル: 皮質モジュール (記憶固定化修正版)
# 目的: ワーキングメモリから長期記憶(RAG)への知識転送を型安全に行う。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging
from .rag_snn import RAGSystem

logger = logging.getLogger(__name__)

class Cortex(nn.Module):
    """
    大脳皮質を模した長期記憶保持・検索モジュール。
    """
    def __init__(self, rag_system: Optional[RAGSystem] = None):
        super().__init__()
        # RAGシステムが注入されない場合は新規作成
        self.rag_system = rag_system or RAGSystem()

    def retrieve(self, query_vector: torch.Tensor) -> List[str]:
        """ベクトルクエリに基づき関連知識を検索。"""
        # ベクトルを文字列クエリに変換（ここでは簡易的に特徴の要約を検索）
        query_str = f"feature_vector_{torch.mean(query_vector).item():.2f}"
        return self.rag_system.search(query_str, k=3)

    def consolidate_memory(self, concept: str, definition: str, importance: float = 1.0):
        """
        [mypy修正] 知識の固定化。
        RAGSystem.update_knowledge ではなく、add_triple を使用する。
        """
        logger.info(f"🧠 Consolidating memory: {concept}")
        
        # 属性エラー回避のため add_triple または add_knowledge を使用
        metadata = {"importance": importance, "type": "consolidated_knowledge"}
        
        # トリプル形式 (主語, 述語, 目的語) で保存
        self.rag_system.add_triple(
            subj=concept, 
            pred="is_defined_as", 
            obj=definition, 
            metadata=metadata
        )

    def get_all_knowledge(self) -> List[str]:
        """全知識のリストを取得。"""
        return self.rag_system.knowledge_base