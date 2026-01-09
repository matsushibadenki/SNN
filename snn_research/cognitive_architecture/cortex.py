# ファイルパス: snn_research/cognitive_architecture/cortex.py
# 日本語タイトル: Cortex v2.1 (Phase 2: Consolidation Interface)
# 目的: 睡眠時の記憶固定化を受け入れる汎用的なインターフェースを追加。

import torch
import torch.nn as nn
from typing import Optional, List
import logging
from .rag_snn import RAGSystem

logger = logging.getLogger(__name__)


class Cortex(nn.Module):
    """
    大脳皮質を模した長期記憶保持・検索モジュール。
    RAGシステムをバックエンドに持ち、意味的知識の貯蔵庫として機能する。
    """

    def __init__(self, rag_system: Optional[RAGSystem] = None):
        super().__init__()
        # RAGシステムが注入されない場合は新規作成
        self.rag_system = rag_system or RAGSystem()
        logger.info("🧠 Cortex initialized (Long-term Knowledge Store).")

    def retrieve(self, query_vector: torch.Tensor) -> List[str]:
        """
        ベクトルクエリに基づき関連知識を検索 (Internal use)。
        """
        # 簡易的にベクトル平均値をキーにする（本来はVector DB検索）
        query_str = f"feature_vector_{torch.mean(query_vector).item():.2f}"
        return self.rag_system.search(query_str, k=3)

    def retrieve_knowledge(self, query: str, k: int = 3) -> List[str]:
        """
        文字列クエリに基づき関連知識を検索 (External/Cognitive use)。
        """
        return self.rag_system.search(query, k=k)

    def consolidate_episode(self, episode_text: str, source: str = "hippocampus"):
        """
        [New] エピソード記憶（テキスト）を長期記憶として保存する。
        SleepConsolidatorから呼び出される。

        Args:
            episode_text: 記憶する内容
            source: 情報源
        """
        try:
            # RAGに追加
            self.rag_system.add_knowledge(episode_text)
            logger.debug(
                f"🧠 Cortex consolidated memory from {source}: {episode_text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to consolidate episode: {e}")

    def consolidate_memory(self, concept: str, definition: str, importance: float = 1.0):
        """
        概念的な知識の固定化（構造化データ）。
        """
        logger.info(f"🧠 Consolidating concept: {concept}")

        metadata = {"importance": importance, "type": "consolidated_concept"}

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
