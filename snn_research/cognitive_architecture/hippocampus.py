# ファイルパス: snn_research/cognitive_architecture/hippocampus.py
# Title: Hippocampal Formation v3.2 (Phase 2: Sleep Support)
# Description: 長期記憶転送(Consolidation)のためのインターフェースを強化。

import logging
import torch
import torch.nn.functional as F
from typing import List, Any, Optional
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
        self.usage = torch.zeros(capacity)  # 使用頻度
        self.pointer = 0
        self.is_full = False

    def store(self, vector: torch.Tensor):
        """
        ベクトルを記憶する。Centroid Updateを行う。
        """
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)

        # Bipolar Transform & Normalize
        # 0.5中心に[-1, 1]へ変換して正規化
        vec_bipolar = (vector - 0.5) * 2.0
        vec_norm = F.normalize(vec_bipolar, p=2, dim=1)

        # 最も似ているスロットを探す
        sim = torch.matmul(vec_norm, self.memory.t()).squeeze(0)

        # マスク: まだ埋まっていないスロットとの類似度は無視（初期値0とのマッチング防止）
        if not self.is_full:
            # 現在のpointer以降は無効とする（簡易的なマスク）
            # 厳密には一度も書き込まれていない場所を管理すべきだが、今回はzeros初期化を利用
            pass

        max_sim, idx = sim.max(dim=0)
        learning_rate = 0.1

        # 類似度閾値 (0.85) 以上なら同一記憶とみなして更新
        if max_sim > 0.85 and (self.is_full or idx < self.pointer):
            current_mem = self.memory[idx]
            # EMA Update: mem = mem + lr * (input - mem)
            new_mem = current_mem + learning_rate * \
                (vec_norm.squeeze(0) - current_mem)
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

        # 未使用領域のスコアを下げる
        if not self.is_full:
            sim[:, self.pointer:] = -2.0  # Cosine sim range is [-1, 1]

        top_v, top_i = sim.topk(k, dim=1)
        return top_i.squeeze(0).tolist()


class Hippocampus:
    """
    短期記憶（STM）とエピソード記憶の一時保管を担当。
    睡眠サイクルにおいて、ここにある記憶がCortex（長期記憶）へ転送される。
    """

    def __init__(
        self,
        rag_system: Optional[RAGSystem] = None,
        short_term_capacity: int = 50,
        working_memory_dim: int = 256
    ):
        self.rag = rag_system if rag_system else RAGSystem()
        # 短期エピソードバッファ (FIFO)
        self.episodic_buffer: deque = deque(maxlen=short_term_capacity)

        # SCAL Associative Memory (ベクトル用短期記憶)
        self.associative_memory = BipolarAssociativeMemory(
            working_memory_dim, capacity=short_term_capacity)

        # ワーキングメモリ (現在の文脈ベクトル)
        self.working_memory = torch.zeros(working_memory_dim)

        logger.info(
            "🧠 Hippocampus initialized (SCAL Memory & Sleep Support Enabled).")

    def process(self, input_data: Any) -> Any:
        """
        入力データの処理と記憶。
        """
        # クエリ処理
        if isinstance(input_data, str) and input_data.startswith("QUERY:"):
            query = input_data.replace("QUERY:", "").strip()
            return self.recall(query)

        # エピソードとして保存
        self.store_episode(input_data)

        # ベクトルが含まれていれば連想記憶へ保存
        if isinstance(input_data, dict) and 'embedding' in input_data:
            emb = input_data['embedding']
            if isinstance(emb, torch.Tensor):
                self.associative_memory.store(emb)
                self.working_memory = emb  # WM更新

        return None

    def store_episode(self, data: Any):
        """短期記憶バッファへ追加"""
        # タイムスタンプなどを付与すると尚良しだが、ここではデータをそのまま保持
        self.episodic_buffer.append(data)

    def recall(self, query: str, k: int = 3) -> List[str]:
        """
        短期記憶およびRAGから情報を検索
        """
        results = []

        # 1. STM Search (直近のバッファからキーワード検索)
        stm_hits = 0
        for item in reversed(self.episodic_buffer):
            item_text = str(item)
            if query in item_text:
                results.append(f"[STM] {item_text[:200]}...")
                stm_hits += 1
                if stm_hits >= 2:
                    break

        # 2. RAG Search (長期記憶)
        if self.rag:
            try:
                rag_results = self.rag.search(query, k=k)
                if rag_results:
                    results.extend(rag_results)
            except Exception:
                pass

        return results

    def flush_memories(self) -> List[Any]:
        """
        [Sleep Cycle用]
        バッファ内の全エピソードを取り出し、バッファをクリアする。
        これは睡眠時の記憶固定化(Consolidation)プロセスで呼ばれることを想定。
        """
        memories = list(self.episodic_buffer)
        self.episodic_buffer.clear()
        logger.info(
            f"🧠 Hippocampus flushed {len(memories)} memories for sleep consolidation.")
        return memories

    def consolidate_memory(self):
        """
        [Legacy/Manual] 手動での固定化呼び出し用。
        通常は SleepConsolidator 経由で行うことを推奨。
        """
        if not self.episodic_buffer:
            return

        items_to_store = self.flush_memories()

        if items_to_store and self.rag:
            # 文字列化して結合保存 (簡易的)
            texts = [str(item) for item in items_to_store]
            combined_text = "\n".join(texts)
            try:
                self.rag.add_knowledge(combined_text)
                logger.info(f"✅ Manually consolidated {len(texts)} episodes.")
            except Exception:
                pass
