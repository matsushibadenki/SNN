# ファイルパス: snn_research/cognitive_architecture/curiosity_knowledge_integrator.py
# 日本語タイトル: Curiosity-Knowledge Graph Integrator v1.0
# 目的・内容:
#   ROADMAP Phase 2.1 「獲得した知識の知識グラフへの統合」を実装。
#   好奇心ドリブン検索(IntrinsicMotivation)で得た情報を、
#   知識グラフ(RAGSystem)およびNeuroSymbolicBridgeへ自動統合する。

import logging
import re
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class AcquiredKnowledge:
    """好奇心で獲得した知識エントリ"""
    query: str                          # 検索クエリ
    content: str                        # 取得したコンテンツ
    source: str = "web_search"          # 情報源
    surprise_score: float = 0.0         # 好奇心スコア
    entities: List[str] = field(default_factory=list)      # 抽出されたエンティティ
    relations: List[tuple] = field(default_factory=list)   # 抽出された関係 (s, p, o)
    embedding: Optional[torch.Tensor] = None               # 埋め込みベクトル


class CuriosityKnowledgeIntegrator:
    """
    好奇心モジュールと知識グラフのブリッジ。
    新しい知識を獲得した際に自動的にグラフへ統合する。
    """

    def __init__(
        self,
        rag_system: Optional[Any] = None,
        neuro_symbolic_bridge: Optional[Any] = None,
        entity_extractor: Optional[Callable[[str], List[str]]] = None,
        relation_extractor: Optional[Callable[[str], List[tuple]]] = None,
        min_surprise_threshold: float = 0.3,  # この閾値以上の驚きがある知識のみ登録
        max_pending_knowledge: int = 100      # 睡眠前にバッファリングする最大数
    ):
        self.rag = rag_system
        self.nsb = neuro_symbolic_bridge  # NeuroSymbolicBridge

        # エンティティ/関係抽出器（デフォルトは簡易ルールベース）
        self.entity_extractor = entity_extractor or self._default_entity_extractor
        self.relation_extractor = relation_extractor or self._default_relation_extractor

        self.min_surprise_threshold = min_surprise_threshold
        self.max_pending_knowledge = max_pending_knowledge

        # 未統合の知識バッファ（睡眠時にまとめて処理）
        self.pending_knowledge: List[AcquiredKnowledge] = []

        # 統計
        self.stats = {
            "total_acquired": 0,
            "total_integrated": 0,
            "total_discarded": 0
        }

        logger.info("🔗 CuriosityKnowledgeIntegrator initialized.")

    def on_knowledge_acquired(
        self,
        query: str,
        content: str,
        surprise_score: float,
        source: str = "web_search"
    ) -> Optional[AcquiredKnowledge]:
        """
        好奇心モジュールが新しい知識を獲得した際のコールバック。

        Args:
            query: 検索クエリ
            content: 取得したコンテンツ
            surprise_score: 好奇心システムが計算した驚きスコア
            source: 情報源（web_search, dialogue, observation等）

        Returns:
            処理されたAcquiredKnowledge、または閾値未満でNone
        """
        self.stats["total_acquired"] += 1

        # 閾値チェック
        if surprise_score < self.min_surprise_threshold:
            self.stats["total_discarded"] += 1
            logger.debug(
                f"📉 Knowledge discarded (surprise={surprise_score:.2f} < threshold={self.min_surprise_threshold})")
            return None

        # エンティティと関係を抽出
        entities = self.entity_extractor(content)
        relations = self.relation_extractor(content)

        knowledge = AcquiredKnowledge(
            query=query,
            content=content,
            source=source,
            surprise_score=surprise_score,
            entities=entities,
            relations=relations
        )

        # 即時統合モード（RAGが利用可能な場合）
        if self.rag is not None:
            self._integrate_to_rag(knowledge)

        # NeuroSymbolicBridgeへのグラウンディングはバッファリング
        if len(self.pending_knowledge) < self.max_pending_knowledge:
            self.pending_knowledge.append(knowledge)
        else:
            # バッファがいっぱいなら最も驚きの低いものを削除
            self.pending_knowledge.sort(key=lambda k: k.surprise_score)
            if knowledge.surprise_score > self.pending_knowledge[0].surprise_score:
                self.pending_knowledge.pop(0)
                self.pending_knowledge.append(knowledge)
                logger.debug("📤 Replaced low-surprise knowledge in buffer.")

        logger.info(
            f"✨ Knowledge acquired: '{query[:30]}...' "
            f"(entities={len(entities)}, relations={len(relations)}, surprise={surprise_score:.2f})")

        return knowledge

    def _integrate_to_rag(self, knowledge: AcquiredKnowledge):
        """知識をRAGシステムに統合"""
        if self.rag is None:
            return

        # メインコンテンツを追加
        metadata = {
            "type": "curiosity_acquired",
            "query": knowledge.query,
            "source": knowledge.source,
            "surprise": knowledge.surprise_score,
            "entities": knowledge.entities
        }
        self.rag.add_knowledge(knowledge.content, metadata=metadata)

        # 抽出された関係をトリプルとして追加
        for rel in knowledge.relations:
            if len(rel) == 3:
                subj, pred, obj = rel
                self.rag.add_triple(subj, pred, obj, metadata={
                                    "source": "curiosity"})

        self.stats["total_integrated"] += 1

    def integrate_during_sleep(self) -> Dict[str, Any]:
        """
        睡眠サイクル中に呼び出される統合処理。
        バッファリングされた知識をNeuroSymbolicBridgeと連携して処理。

        Returns:
            統合レポート
        """
        if not self.pending_knowledge:
            return {"status": "no_pending_knowledge", "integrated": 0}

        integrated_count = 0
        grounded_concepts = []

        for knowledge in self.pending_knowledge:
            # 1. エンティティをSNNパターンにグラウンディング
            if self.nsb is not None:
                for entity in knowledge.entities:
                    try:
                        self.nsb.ground_symbol(entity)  # グラウンディング実行
                        grounded_concepts.append(entity)
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to ground entity '{entity}': {e}")

            # 2. 関係をSNN結合強化として反映
            if self.nsb is not None and hasattr(self.nsb, 'learn_from_dialogue'):
                # 簡易的なスパイクパターン生成
                import numpy as np
                dummy_pattern = np.random.randn(256)
                self.nsb.learn_from_dialogue(
                    knowledge.content[:500], dummy_pattern)

            integrated_count += 1

        # バッファクリア
        self.pending_knowledge.clear()

        report = {
            "status": "success",
            "integrated": integrated_count,
            "grounded_concepts": grounded_concepts,
            "total_stats": self.stats.copy()
        }

        logger.info(
            f"🛌 Sleep integration complete: {integrated_count} knowledge entries processed, "
            f"{len(grounded_concepts)} concepts grounded.")

        return report

    def _default_entity_extractor(self, text: str) -> List[str]:
        """
        簡易的なエンティティ抽出（大文字で始まる単語）
        実運用では spaCy や Transformerベースの NER を使用推奨
        """
        # 大文字で始まる単語を抽出
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # 重複除去しつつ順序保持
        seen = set()
        entities = []
        for w in words:
            if w not in seen and len(w) > 2:
                seen.add(w)
                entities.append(w)
        return entities[:20]  # 最大20エンティティ

    def _default_relation_extractor(self, text: str) -> List[tuple]:
        """
        簡易的な関係抽出 (パターンマッチング)
        実運用では OpenIE や KG構築ツールを使用推奨
        """
        relations = []

        # "X is Y" パターン
        is_pattern = re.findall(
            r'(\b[A-Z][a-z]+\b)\s+is\s+(?:a|an|the)?\s*(\b[a-z]+\b)', text)
        for subj, obj in is_pattern:
            relations.append((subj, "is_a", obj))

        # "X causes Y" パターン
        cause_pattern = re.findall(
            r'(\b[A-Z][a-z]+\b)\s+causes?\s+(\b[a-z]+\b)', text, re.IGNORECASE)
        for subj, obj in cause_pattern:
            relations.append((subj, "causes", obj))

        # "X has Y" パターン
        has_pattern = re.findall(
            r'(\b[A-Z][a-z]+\b)\s+has\s+(?:a|an)?\s*(\b[a-z]+\b)', text)
        for subj, obj in has_pattern:
            relations.append((subj, "has", obj))

        return relations[:10]  # 最大10関係

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return {
            **self.stats,
            "pending_count": len(self.pending_knowledge)
        }


# --- Factory Function for Brain Integration ---
def create_curiosity_integrator(
    rag_system: Optional[Any] = None,
    neuro_symbolic_bridge: Optional[Any] = None
) -> CuriosityKnowledgeIntegrator:
    """
    ブレインシステムから利用するためのファクトリ関数。
    """
    return CuriosityKnowledgeIntegrator(
        rag_system=rag_system,
        neuro_symbolic_bridge=neuro_symbolic_bridge
    )
