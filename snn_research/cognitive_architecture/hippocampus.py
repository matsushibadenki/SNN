# ファイルパス: snn_research/cognitive_architecture/hippocampus.py
# Title: Hippocampal Formation v3.0 (SCAL Associative Memory)
# Description: Bipolar Centroid Learningによる堅牢なベクトル連想記憶を追加。

import logging
import torch
import torch.nn.functional as F
import json
from typing import List, Dict, Any, Optional, Union
from collections import deque
from snn_research.cognitive_architecture.rag_snn import RAGSystem

logger = logging.getLogger(__name__)

class BipolarAssociativeMemory:
    """
    SCALに基づく連想記憶マトリクス。
    ベクトルをバイポーラ重心学習で記憶し、コサイン類似度で想起する。
    """
    def __init__(self, dim: int, capacity: int = 100):
        self.dim = dim
        self.capacity = capacity
        # 記憶マトリクス: [Capacity, Dim]
        # 正規化されたプロトタイプベクトルを保持
        self.memory = torch.zeros(capacity, dim)
        self.usage = torch.zeros(capacity) # 使用頻度またはLRU用
        self.pointer = 0
        self.is_full = False

    def store(self, vector: torch.Tensor):
        """
        ベクトルを記憶する。Centroid Updateを行う。
        """
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        
        # Bipolar Transform & Normalize
        vec_bipolar = (vector - 0.5) * 2.0
        vec_norm = F.normalize(vec_bipolar, p=2, dim=1)
        
        # 最も似ているスロットを探す (閾値以上なら更新、なければ新規)
        sim = torch.matmul(vec_norm, self.memory.t()).squeeze(0)
        max_sim, idx = sim.max(dim=0)
        
        learning_rate = 0.1
        
        if max_sim > 0.8: # 既存記憶の更新 (Centroid Averaging)
            current_mem = self.memory[idx]
            # EMA Update: mem = mem + lr * (input - mem)
            new_mem = current_mem + learning_rate * (vec_norm.squeeze(0) - current_mem)
            self.memory[idx] = F.normalize(new_mem, p=2, dim=0)
            self.usage[idx] += 1
        else:
            # 新規書き込み (Ring Buffer)
            target_idx = self.pointer
            self.memory[target_idx] = vec_norm.squeeze(0)
            self.usage[target_idx] = 1
            
            self.pointer = (self.pointer + 1) % self.capacity
            if self.pointer == 0:
                self.is_full = True

    def retrieve(self, query_vector: torch.Tensor, k: int = 1) -> List[int]:
        """
        類似メモリのインデックスを返す。
        """
        if query_vector.dim() == 1:
            query_vector = query_vector.unsqueeze(0)
            
        q_bipolar = (query_vector - 0.5) * 2.0
        q_norm = F.normalize(q_bipolar, p=2, dim=1)
        
        sim = torch.matmul(q_norm, self.memory.t())
        # マスクしていない領域(未書き込み)は除外
        if not self.is_full:
            sim[:, self.pointer:] = -1.0
            
        top_v, top_i = sim.topk(k, dim=1)
        return top_i.squeeze(0).tolist()

class Hippocampus:
    def __init__(
        self,
        rag_system: Optional[RAGSystem] = None,
        short_term_capacity: int = 50,
        working_memory_dim: int = 256
    ):
        self.rag = rag_system if rag_system else RAGSystem()
        self.episodic_buffer: deque = deque(maxlen=short_term_capacity)
        
        # SCAL Associative Memory
        self.associative_memory = BipolarAssociativeMemory(working_memory_dim, capacity=short_term_capacity)
        self.working_memory = torch.zeros(working_memory_dim)
        
        logger.info("🧠 Hippocampus initialized (SCAL Memory Enabled).")

    def process(self, input_data: Any) -> Any:
        if isinstance(input_data, str) and input_data.startswith("QUERY:"):
            query = input_data.replace("QUERY:", "").strip()
            return self.recall(query)
        
        # データ保存と同時にベクトル学習も行う(簡易シミュレーション)
        self.store_episode(input_data)
        
        # もしinput_dataがテンソルやベクトルを持っていれば連想記憶へ
        if isinstance(input_data, dict) and 'embedding' in input_data:
            emb = input_data['embedding']
            if isinstance(emb, torch.Tensor):
                self.associative_memory.store(emb)
                
        return None

    def store_episode(self, data: Any):
        self.episodic_buffer.append(data)

    def recall(self, query: str, k: int = 3) -> List[str]:
        results = []
        
        # 1. STM Search (Text)
        stm_hits = 0
        for item in reversed(self.episodic_buffer):
            item_text = str(item)
            if query in item_text:
                results.append(f"[STM] {item_text[:200]}...")
                stm_hits += 1
                if stm_hits >= 2: break 

        # 2. RAG Search
        if self.rag:
            try:
                rag_results = self.rag.search(query, k=k)
                if rag_results:
                    results.extend(rag_results)
            except Exception:
                pass
            
        return results

    def consolidate_memory(self):
        """STM -> LTM Transfer"""
        if not self.episodic_buffer:
            return
        
        items_to_store = []
        while self.episodic_buffer:
            item = self.episodic_buffer.popleft()
            if isinstance(item, str):
                items_to_store.append(item)
            elif isinstance(item, dict):
                items_to_store.append(str(item))
        
        if items_to_store and self.rag:
            combined_text = "\n".join(items_to_store)
            try:
                self.rag.add_knowledge(combined_text)
                logger.info(f"✅ Consolidated {len(items_to_store)} episodes.")
            except Exception:
                pass
    
    def integrate_knowledge(self, topic: str, source_path: str):
        pass
