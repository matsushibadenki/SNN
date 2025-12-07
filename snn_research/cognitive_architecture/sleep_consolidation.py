# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# Title: 睡眠時記憶固定化システム (Generative Replay & Consolidation) v2.1
# Description:
#   Neuro-Symbolic Feedback Loopの実装。
#   GraphRAGの知識をサンプリングし、SNNへの感覚入力として「夢」を生成・再生する。
#   Causal Trace Learningを用いて、エピソード記憶を長期的なシナプス重みに焼き付ける。

import torch
import logging
import random
import time
from typing import List, Dict, Any, Optional, cast, Union
import numpy as np

from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.core.base import BaseModel

logger = logging.getLogger(__name__)

class SleepConsolidator:
    """
    睡眠サイクルにおいて、知識の構造化（Symbolic）とニューラルネットワークへの固定化（Sub-symbolic）を行う。
    """
    def __init__(
        self, 
        rag_system: RAGSystem, 
        cortex_snn: Union[BaseModel, torch.nn.Module], 
        spike_encoder: SpikeEncoder,
        consolidation_epochs: int = 3,
        replay_batch_size: int = 4,
        synaptic_scaling_factor: float = 0.98 # SHY仮説に基づくダウンスケーリング係数
    ):
        self.rag_system = rag_system
        self.cortex_snn = cortex_snn
        self.spike_encoder = spike_encoder
        self.consolidation_epochs = consolidation_epochs
        self.replay_batch_size = replay_batch_size
        self.synaptic_scaling_factor = synaptic_scaling_factor
        
        logger.info("💤 SleepConsolidator initialized. Ready to dream.")

    def _generate_dream_content(self, limit: int = 20) -> List[str]:
        """
        GraphRAGから「夢」のコンテンツを生成する。
        重要度（次数中心性）と新鮮さ（最近追加された知識）に基づいてサンプリングを行う。
        """
        if not self.rag_system.knowledge_graph or self.rag_system.knowledge_graph.number_of_nodes() == 0:
            return ["Empty void."] # 知識がない場合のデフォルト夢
            
        graph = self.rag_system.knowledge_graph
        nodes = list(graph.nodes())
        
        # サンプリング戦略: 次数中心性による重み付け（重要な概念ほど夢に出やすい）
        degrees = [val for (node, val) in graph.degree()]
        total_degree = sum(degrees)
        
        if total_degree == 0:
            probs = None
        else:
            probs = [d / total_degree for d in degrees]
            
        # ランダムサンプリング (夢の非線形性・跳躍を模倣)
        try:
            selected_nodes = np.random.choice(
                nodes, 
                size=min(len(nodes), limit), 
                p=probs, 
                replace=False
            ).tolist()
        except ValueError:
            selected_nodes = nodes[:limit]
        
        dream_texts = []
        for node in selected_nodes:
            # その概念周辺のサブグラフ情報を自然言語化して「夢のシーン」を作る
            # 例: "Cat is a Animal. Cat has whiskers."
            info_list = self.rag_system.get_subgraph_info(node)
            if info_list:
                # 複数の関係性を結合
                scene = ". ".join(info_list[:3]) + "."
                dream_texts.append(scene)
                
        return dream_texts

    def perform_sleep_cycle(self) -> Dict[str, Any]:
        """
        睡眠サイクルを実行: 夢の生成 -> SNNでの再生 -> シナプス調整
        """
        start_time = time.time()
        
        # 1. 夢の生成 (Retrieval from Symbolic Memory)
        dream_contents = self._generate_dream_content(limit=12)
        if not dream_contents:
            return {"synaptic_change": 0.0, "duration": 0.0, "dreams_replayed": 0}

        # 2. ニューラル・リプレイ (Generative Replay on SNN)
        self.cortex_snn.train() # SNNを学習モードに移行
        
        device = torch.device("cpu")
        try:
            device = next(self.cortex_snn.parameters()).device
        except StopIteration: pass
        
        # モデルのタイムステップ取得
        time_steps = 16 # Default
        if hasattr(self.cortex_snn, 'time_steps'): 
             pass 
        
        total_synaptic_change = 0.0

        for epoch in range(self.consolidation_epochs):
            batch_change = 0.0
            
            # 夢をバッチ処理
            for i in range(0, len(dream_contents), self.replay_batch_size):
                batch_texts = dream_contents[i : i + self.replay_batch_size]
                
                # --- Symbol to Spike (記号接地) ---
                batch_spikes_list = []
                for text in batch_texts:
                    # テキスト内容をスパイク列にエンコード
                    # これにより、言語的な知識が神経パルスに変換される
                    spikes = self.spike_encoder.encode(
                        {"content": text, "type": "text"}, 
                        duration=time_steps
                    )
                    batch_spikes_list.append(spikes)
                
                if not batch_spikes_list: continue
                
                # (Batch, Time, Neurons)
                input_tensor = torch.stack(batch_spikes_list).to(device)
                
                # --- SNN Plasticity Update ---
                if hasattr(self.cortex_snn, 'reset_state'):
                    self.cortex_snn.reset_state() # type: ignore

                # 順伝播: 夢を見る (ニューロン活動の誘起)
                # CorticalColumnのforwardは (input, prev_states) を取る場合があるため、
                # 柔軟に対応する
                try:
                    _ = self.cortex_snn(input_tensor)
                except TypeError:
                    # 引数が合わない場合のフォールバック（状態なし呼び出し）
                    _ = self.cortex_snn(input_tensor, None)
                
                # 学習則の適用 (run_learning_step)
                # 夢見学習ではターゲットなしの自己教師あり学習（Hebbian/STDP）を行う
                if hasattr(self.cortex_snn, 'run_learning_step'):
                    metrics = self.cortex_snn.run_learning_step( # type: ignore
                        inputs=input_tensor, 
                        targets=None
                    )
                    
                    # 更新量の集計
                    for k, v in metrics.items():
                        if "magnitude" in k or "update" in k:
                            val = v.item() if isinstance(v, torch.Tensor) else float(v)
                            batch_change += val
                else:
                    logger.warning("Cortex SNN does not support 'run_learning_step'. Plasticity skipped.")

            total_synaptic_change += batch_change

        # 3. Synaptic Homeostasis (SHY仮説)
        # 覚醒中に増強されたシナプスを全体的にダウンスケーリングし、重要な記憶だけを際立たせる
        self._apply_synaptic_scaling()
        
        duration = time.time() - start_time
        
        logger.info(f"💤 Sleep cycle finished. Total synaptic update magnitude: {total_synaptic_change:.4f}")
        
        return {
            "synaptic_change": total_synaptic_change, 
            "duration": duration,
            "dreams_replayed": len(dream_contents)
        }

    def _apply_synaptic_scaling(self):
        """シナプス重みの全体的なダウンスケーリング"""
        with torch.no_grad():
            for param in self.cortex_snn.parameters():
                if param.requires_grad and param.dim() > 1: # 重み行列のみ
                    param.mul_(self.synaptic_scaling_factor)
