# /snn_research/cognitive_architecture/rag_snn.py
# 日本語タイトル: Spiking RAG System (インターフェース整合版)
# 目的: 因果推論エンジンからの strength 引数を受け入れ可能にし、mypyエラーを解消する。

import logging
import torch
import os
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, embedding_dim: Optional[int] = None, vector_store_path: Optional[str] = None):
        self.knowledge_base: List[str] = []
        self.metadata_store: List[Dict[str, Any]] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.vector_store_path = vector_store_path
        
        if self.vector_store_path:
            try:
                os.makedirs(self.vector_store_path, exist_ok=True)
                logger.info(f"📁 Vector store directory ready at: {self.vector_store_path}")
            except Exception as e:
                logger.error(f"❌ Failed to create vector store directory: {e}")

        self.has_encoder = False
        self.encoder: Any = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_encoder = True
            detected_dim = self.encoder.get_sentence_embedding_dimension()
            self.embedding_dim = int(detected_dim) if isinstance(detected_dim, int) else 384
        except ImportError:
            logger.warning("⚠️ sentence_transformers not found. Using random embeddings.")
            self.embedding_dim = embedding_dim if embedding_dim is not None else 768

    def _encode(self, texts: List[str]) -> torch.Tensor:
        if self.has_encoder and self.encoder:
            embeddings = self.encoder.encode(texts, convert_to_tensor=True)
            return embeddings.cpu()
        return torch.randn(len(texts), self.embedding_dim)

    def add_knowledge(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        chunks = [text] 
        new_vecs = self._encode(chunks)
        self.knowledge_base.extend(chunks)
        self.metadata_store.append(metadata if metadata else {})

        if self.embeddings is None:
            self.embeddings = new_vecs
        else:
            self.embeddings = torch.cat([self.embeddings, new_vecs], dim=0)

    # --- 因果関係の結晶化 (修正箇所) ---
    def add_causal_relationship(
        self, 
        cause: str, 
        effect: str, 
        strength: float = 1.0, 
        condition: Optional[str] = None
    ):
        """
        [Fix] 因果関係を追加。引数名を strength に変更し CausalInferenceEngine と整合。
        """
        if condition:
            text_rep = f"If {condition}, because {cause}, then {effect}. (Strength: {strength:.2f})"
        else:
            text_rep = f"Because {cause}, then {effect}. (Strength: {strength:.2f})"
            
        self.add_knowledge(text_rep, metadata={
            "type": "causal", 
            "cause": cause, 
            "effect": effect, 
            "strength": strength, 
            "condition": condition
        })

    def add_triple(self, subj: str, pred: str, obj: str, metadata: Optional[Dict[str, Any]] = None):
        text_rep = f"{subj} {pred} {obj}"
        meta = metadata if metadata else {}
        meta.update({"type": "triple", "subject": subj, "predicate": pred, "object": obj})
        self.add_knowledge(text_rep, metadata=meta)

    def search(self, query: str, k: int = 3) -> List[str]:
        if self.embeddings is None or len(self.knowledge_base) == 0:
            return []
        query_vec = self._encode([query])
        db_norm = F.normalize(self.embeddings, p=2, dim=1)
        q_norm = F.normalize(query_vec, p=2, dim=1)
        scores = torch.mm(q_norm, db_norm.transpose(0, 1)).squeeze(0)
        top_k = min(k, len(self.knowledge_base))
        if top_k == 0: return []
        indices = torch.topk(scores, k=top_k).indices
        return [self.knowledge_base[int(i)] for i in indices]