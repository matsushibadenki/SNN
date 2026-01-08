# ファイルパス: snn_research/cognitive_architecture/cortex.py
# 日本語タイトル: 皮質モジュール (Knowledge Retrieval Fix)
# 目的: 文字列ベースの知識検索メソッド(retrieve_knowledge)を追加し、Brain v14デモのエラーを解消する。

import torch
import torch.nn as nn
from typing import Optional, List
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
        """
        ベクトルクエリに基づき関連知識を検索。
        主にArtificialBrainの内部処理(perceptual_info経由)で使用。
        """
        # ベクトルを文字列クエリに変換（ここでは簡易的に特徴の要約を検索）
        # 実用上はVector Storeの検索APIにベクトルを直接渡すが、
        # 現在のRAGSystemは文字列検索ベースのため、疑似的なキーを生成
        query_str = f"feature_vector_{torch.mean(query_vector).item():.2f}"
        return self.rag_system.search(query_str, k=3)

    def retrieve_knowledge(self, query: str, k: int = 3) -> List[str]:
        """
        [Fix] 文字列クエリに基づき関連知識を検索。
        Brain v14シナリオ等の高次認知プロセスから直接呼び出される。
        """
        return self.rag_system.search(query, k=k)

    def consolidate_memory(self, concept: str, definition: str, importance: float = 1.0):
        """
        知識の固定化。
        RAGSystemにトリプル形式で知識を追加する。
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