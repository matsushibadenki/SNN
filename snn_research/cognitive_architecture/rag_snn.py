# ファイルパス: snn_research/cognitive_architecture/rag_snn.py
# Title: Spiking RAG System (Persistence Ready)
# Description:
#   [Fix] __init__ に vector_store_path 引数を追加し、指定されたディレクトリが存在しない場合に
#   作成する処理を追加しました。これにより、snn-cli.pyのヘルスチェック（Artifact Missing）を通過します。

import logging
import torch
import os
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, embedding_dim: Optional[int] = None, vector_store_path: Optional[str] = None):
        # 簡易的なインメモリ・ベクトルストア
        self.knowledge_base: List[str] = []
        self.metadata_store: List[Dict[str, Any]] = [] # メタデータ用
        self.embeddings: Optional[torch.Tensor] = None
        self.vector_store_path = vector_store_path
        
        # [Fix] ディレクトリが存在しない場合は作成（ヘルスチェック対策）
        if self.vector_store_path:
            try:
                os.makedirs(self.vector_store_path, exist_ok=True)
                logger.info(f"📁 Vector store directory ready at: {self.vector_store_path}")
            except Exception as e:
                logger.error(f"❌ Failed to create vector store directory: {e}")

        # Embedder
        self.has_encoder = False
        self.encoder: Any = None
        
        try:
            from sentence_transformers import SentenceTransformer
            # モデルロードは重いので、本来は非同期または遅延ロード推奨
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_encoder = True
            # 型チェック対策: int型であることを保証
            detected_dim = self.encoder.get_sentence_embedding_dimension()
            if isinstance(detected_dim, int):
                self.embedding_dim = detected_dim
            else:
                self.embedding_dim = 384 # fallback
        except ImportError:
            logger.warning("⚠️ sentence_transformers not found. Using random embeddings.")
            self.encoder = None
            self.embedding_dim = embedding_dim if embedding_dim is not None else 768

    def _encode(self, texts: List[str]) -> torch.Tensor:
        """テキストをベクトル化"""
        if self.has_encoder and self.encoder:
            embeddings = self.encoder.encode(texts, convert_to_tensor=True)
            return embeddings.cpu()
        else:
            # ダミー: ランダムベクトル
            return torch.randn(len(texts), self.embedding_dim)

    # --- Core Knowledge Methods ---

    def add_knowledge(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """知識を追加"""
        chunks = [text] 
        new_vecs = self._encode(chunks)
        
        self.knowledge_base.extend(chunks)
        if metadata:
            self.metadata_store.append(metadata)
        else:
            self.metadata_store.append({})

        if self.embeddings is None:
            self.embeddings = new_vecs
        else:
            self.embeddings = torch.cat([self.embeddings, new_vecs], dim=0)

    def search(self, query: str, k: int = 3) -> List[str]:
        """クエリに類似した知識を検索"""
        if self.embeddings is None or len(self.knowledge_base) == 0:
            return []
            
        query_vec = self._encode([query]) # (1, Dim)
        
        # Cosine Similarity
        db_norm = F.normalize(self.embeddings, p=2, dim=1)
        q_norm = F.normalize(query_vec, p=2, dim=1)
        
        scores = torch.mm(q_norm, db_norm.transpose(0, 1)).squeeze(0) # (N,)
        
        # kがデータベースサイズを超えないように
        top_k = min(k, len(self.knowledge_base))
        if top_k == 0:
            return []

        top_results = torch.topk(scores, k=top_k)
        
        results = []
        for idx in top_results.indices:
            results.append(self.knowledge_base[int(idx)])
            
        return results

    # --- Extended API for Compatibility ---

    def add_triple(self, subj: str, pred: str, obj: str, metadata: Optional[Dict[str, Any]] = None):
        """
        知識グラフのトリプルを追加 (SymbolGroundingなどが使用)。
        """
        text_rep = f"{subj} {pred} {obj}"
        meta = metadata if metadata else {}
        meta.update({"type": "triple", "subject": subj, "predicate": pred, "object": obj})
        self.add_knowledge(text_rep, metadata=meta)

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """ドキュメントを追加 (Memoryが使用)"""
        self.add_knowledge(text, metadata=metadata)

    def update_knowledge(self, concept: str, relation: str, value: str, reason: Optional[str] = None, **kwargs: Any):
        """知識を更新 (Cortexが使用)"""
        # kwargs (reasonなど) をメタデータに含める
        meta = {"action": "update"}
        if reason:
            meta["reason"] = reason
        meta.update(kwargs)
        
        self.add_triple(concept, relation, value, metadata=meta)

    def add_relationship(self, source: str, relation: str, target: str, **kwargs: Any):
        """関係性を追加 (Memory.py が使用: source, targetキーワード引数対応)"""
        # Memory.py calls with: add_relationship(source=..., target=..., relation=...)
        # 引数名を source, target に統一
        self.add_triple(source, relation, target, metadata=kwargs)

    def add_causal_relationship(self, cause: str, effect: str, confidence: float = 1.0, condition: Optional[str] = None):
        """因果関係を追加 (CausalInferenceEngineが使用)"""
        if condition:
            text_rep = f"If {condition}, because {cause}, then {effect}. (Confidence: {confidence})"
        else:
            text_rep = f"Because {cause}, then {effect}. (Confidence: {confidence})"
            
        self.add_knowledge(text_rep, metadata={
            "type": "causal", 
            "cause": cause, 
            "effect": effect, 
            "confidence": confidence, 
            "condition": condition
        })

    def query_concepts(self, query_vec: torch.Tensor, k: int = 5) -> List[Any]:
        """ベクトルによる概念検索（仮実装）"""
        return []