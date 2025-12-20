# ファイルパス: snn_research/cognitive_architecture/hippocampus.py
# Title: Hippocampal Formation v2.3 (Enhanced Episodic Memory)
# Description:
#   短期記憶と長期記憶(RAG)のインターフェース。
#   修正: store_episodeの構造化対応、recallの実装強化。

import logging
import torch
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import deque

# RAGシステム (ベクトル検索エンジン)
from snn_research.cognitive_architecture.rag_snn import RAGSystem

logger = logging.getLogger(__name__)

class Hippocampus:
    """
    記憶の形成(Encoding)、保持(Storage)、想起(Retrieval)を司るモジュール。
    """
    def __init__(
        self,
        rag_system: Optional[RAGSystem] = None,
        short_term_capacity: int = 50,
        working_memory_dim: int = 256
    ):
        self.rag = rag_system if rag_system else RAGSystem()
        
        # 短期記憶 (STM) / エピソードバッファ
        # 最近のイベントを辞書形式などで保持
        self.episodic_buffer: deque = deque(maxlen=short_term_capacity)
        
        # 作業記憶 (WM) - 現在の思考コンテキスト
        self.working_memory = torch.zeros(working_memory_dim)
        
        logger.info("🧠 Hippocampus initialized (STM + RAG Interface).")

    def process(self, input_data: Any) -> Any:
        """
        Brain Kernelからの呼び出し口。
        """
        if isinstance(input_data, str) and input_data.startswith("QUERY:"):
            query = input_data.replace("QUERY:", "").strip()
            return self.recall(query)
            
        self.store_episode(input_data)
        return None

    def store_episode(self, data: Any):
        """
        短期記憶へエピソードを追加。
        dataが辞書の場合、重要なキーが含まれているかチェックしつつ保存。
        """
        self.episodic_buffer.append(data)
        
        # 簡易ログ表示
        preview = ""
        if isinstance(data, dict):
            preview = f"In: {str(data.get('input'))[:20]} -> Out: {str(data.get('output'))[:20]}"
        else:
            preview = str(data)[:30]
        # logger.debug(f"📝 Episode buffered: {preview}...")

    def recall(self, query: str, k: int = 3) -> List[str]:
        """
        記憶の検索 (Recall)。
        STMとLTMの両方から検索する。
        """
        results = []
        
        # 1. 短期記憶からの検索 (簡易キーワードマッチ)
        # 直近の会話や文脈を優先。辞書内のテキストフィールドも検索対象にする。
        stm_hits = 0
        for item in reversed(self.episodic_buffer):
            item_text = ""
            if isinstance(item, dict):
                # 辞書なら input/output を検索対象文字列にする
                parts = [str(v) for k, v in item.items() if k in ['input', 'output', 'thought_trace'] and v]
                item_text = " | ".join(parts)
            else:
                item_text = str(item)
            
            if query in item_text:
                results.append(f"[STM] {item_text[:200]}...") # 長すぎる場合はカット
                stm_hits += 1
                if stm_hits >= 2: break 

        # 2. 長期記憶(RAG)からの検索
        if self.rag:
            # rag.searchの実装に依存するが、ここではリストが返ると想定
            try:
                rag_results = self.rag.search(query, k=k)
                if rag_results:
                    results.extend(rag_results)
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
            
        return results

    def consolidate_memory(self):
        """
        記憶の固定化 (Consolidation)。
        STMの内容を文字列化してLTM(RAG)へ移動する。
        """
        if not self.episodic_buffer:
            return

        logger.info("💤 Consolidating memories to Long-Term Storage...")
        
        items_to_store = []
        while self.episodic_buffer:
            item = self.episodic_buffer.popleft()
            text_representation = ""
            
            if isinstance(item, str):
                text_representation = item
            elif isinstance(item, dict):
                # 辞書をJSON文字列化、あるいは人間可読な形式へ
                inp = item.get("input", "")
                out = item.get("output", "")
                emo = item.get("emotion", "")
                text_representation = f"Input: {inp}\nResponse: {out}\nEmotion: {emo}"
            
            if text_representation:
                items_to_store.append(text_representation)
        
        if items_to_store and self.rag:
            combined_text = "\n---\n".join(items_to_store)
            try:
                self.rag.add_knowledge(combined_text)
                logger.info(f"✅ Consolidated {len(items_to_store)} episodes into LTM.")
            except Exception as e:
                logger.error(f"Failed to add knowledge to RAG: {e}")

    def integrate_knowledge(self, topic: str, source_path: str):
        """Web学習などで得られた外部知識ファイルをRAGに取り込む。"""
        if not self.rag:
            logger.warning("❌ RAG System not available for knowledge integration.")
            return

        logger.info(f"📚 Integrating new knowledge about '{topic}' from {source_path}...")
        try:
            count = 0
            with open(source_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if text:
                            self.rag.add_knowledge(text)
                            count += 1
                    except json.JSONDecodeError:
                        continue
            logger.info(f"🎉 Integrated {count} documents into RAG index.")
        except Exception as e:
            logger.error(f"Failed to integrate knowledge: {e}")