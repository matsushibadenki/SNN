# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# Title: 睡眠時記憶固定化システム (Generative Replay & Consolidation)
# Description:
#   ROADMAP Phase 5 の中核。
#   1. Explicit Consolidation: 海馬（短期記憶）のエピソードをGraphRAG（長期記憶）へ構造化して転送。
#   2. Implicit Consolidation: GraphRAGの知識を「夢」として再構成し、SNN（大脳皮質）へリプレイ入力。
#   3. Synaptic Homeostasis: シナプス重みのスケーリングにより、学習による暴走を防ぐ。
#
#   修正: mypyエラーの解消。
#     - cortex_snn のメソッド呼び出しにおける型エラー ("Tensor" not callable) を修正。
#     - 未定義変数 total_plasticity_change を total_synaptic_change に修正。

import torch
import logging
import random
import time
from typing import List, Dict, Any, Optional, cast, Union
import networkx as nx # type: ignore[import-untyped]
import numpy as np

# 既存モジュールのインポート
from snn_research.cognitive_architecture.rag_snn import RAGSystem
# 循環参照回避のため型ヒントのみ
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.core.base import BaseModel

logger = logging.getLogger(__name__)

class SleepConsolidator:
    """
    睡眠サイクルを管理し、知識の構造化とニューラルネットワークへの焼き付けを行う。
    """
    def __init__(
        self, 
        rag_system: RAGSystem, 
        cortex_snn: Any, # 修正: 型をAnyにしてメソッド呼び出しの柔軟性を持たせる (BaseModelだとreset_state等がない)
        spike_encoder: SpikeEncoder,
        consolidation_epochs: int = 3,
        replay_batch_size: int = 4,
        synaptic_scaling_factor: float = 0.9 # ダウン・スケーリング係数
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
        最近活性化したノードや、重要なハブノードを中心に知識をサンプリングする。
        """
        if not self.rag_system.knowledge_graph or self.rag_system.knowledge_graph.number_of_nodes() == 0:
            return []
            
        graph = self.rag_system.knowledge_graph
        nodes = list(graph.nodes())
        
        # 戦略A: ランダムサンプリング（探索的夢）
        # 戦略B: 次数が高いノード（重要な概念）
        # 戦略C: 最近追加されたノード（エピソード記憶）
        
        # 簡易的に次数ベースの重み付けサンプリング
        degrees = [val for (node, val) in graph.degree()]
        total_degree = sum(degrees)
        if total_degree == 0:
            probs = None
        else:
            probs = [d / total_degree for d in degrees]
            
        selected_nodes = np.random.choice(nodes, size=min(len(nodes), limit), p=probs, replace=False).tolist()
        
        dream_texts = []
        for node in selected_nodes:
            # その概念周辺のサブグラフを文章化
            info_list = self.rag_system.get_subgraph_info(node)
            if info_list:
                dream_texts.append(" ".join(info_list))
                
        return dream_texts

    def perform_sleep_cycle(self) -> Dict[str, float]:
        """
        睡眠サイクルを実行する。
        """
        start_time = time.time()
        logger.info("💤 --- Entering Sleep Phase (Consolidation) ---")
        
        # 1. 夢の生成 (Knowledge Retrieval)
        dream_contents = self._generate_dream_content(limit=10)
        if not dream_contents:
            logger.info("   (No knowledge to replay. Sleeping deeply...)")
            return {"synaptic_change": 0.0, "duration": time.time() - start_time}

        logger.info(f"   Generated {len(dream_contents)} dream fragments for replay.")

        # 2. ニューラル・リプレイ (Replay Learning)
        # SNNを学習モードへ
        if isinstance(self.cortex_snn, torch.nn.Module):
            self.cortex_snn.train()
        
        total_synaptic_change = 0.0
        
        # SNNのモデルデバイスを取得
        device = torch.device("cpu")
        if isinstance(self.cortex_snn, torch.nn.Module):
            try:
                device = next(self.cortex_snn.parameters()).device
            except StopIteration:
                pass
            
        # タイムステップの取得
        time_steps = getattr(self.cortex_snn, 'time_steps', 16)

        for epoch in range(self.consolidation_epochs):
            batch_change = 0.0
            
            # バッチごとに処理
            for i in range(0, len(dream_contents), self.replay_batch_size):
                batch_texts = dream_contents[i : i + self.replay_batch_size]
                
                # スパイクエンコーディング (Symbol -> Spike)
                # 夢の内容を感覚入力として再現
                batch_spikes_list = []
                for text in batch_texts:
                    spikes = self.spike_encoder.encode(
                        {"content": text, "type": "text"}, 
                        duration=time_steps
                    )
                    batch_spikes_list.append(spikes)
                
                # スタックしてバッチ化: (Batch, Time, Neurons)
                # encodeの戻り値が (Time, Neurons) 前提
                input_tensor = torch.stack(batch_spikes_list).to(device)
                
                # --- SNN Forward & Plasticity Update ---
                # リセット
                if hasattr(self.cortex_snn, 'reset_state'):
                    # 修正: 明示的にメソッドとして呼び出す (mypy対策でAnyキャスト済み)
                    self.cortex_snn.reset_state()
                
                # 順伝播 (教師なし/自己教師あり学習を想定)
                # BioPCNetwork等の場合、入力自体をターゲットとして予測誤差を最小化する
                # mypyエラー "Tensor" not callable を回避するため Any として扱う
                _ = self.cortex_snn(input_tensor) 
                
                # 学習則の適用 (run_learning_step メソッドを持つことを期待)
                if hasattr(self.cortex_snn, 'run_learning_step'):
                    # BioPCNetworkなどは targets 引数が必要な場合がある
                    metrics = self.cortex_snn.run_learning_step(inputs=input_tensor, targets=input_tensor)
                    
                    # 更新量の集計（ログ用）
                    for k, v in metrics.items():
                        if "magnitude" in k:
                            if isinstance(v, torch.Tensor):
                                batch_change += v.item()
                            elif isinstance(v, (float, int)):
                                batch_change += float(v)
            
            total_synaptic_change += batch_change
            logger.debug(f"   [Sleep Epoch {epoch+1}] Synaptic change: {batch_change:.4f}")

        # 3. シナプス恒常性維持 (Synaptic Scaling)
        # 全シナプスを一律にダウンスケーリングし、飽和を防ぐ (SHY仮説)
        self._apply_synaptic_scaling()
        
        duration = time.time() - start_time
        logger.info(f"💤 Sleep cycle finished in {duration:.2f}s. Total synaptic change: {total_synaptic_change:.4f}")
        
        # 修正: 未定義変数 total_plasticity_change を total_synaptic_change に修正
        return {
            "synaptic_change": total_synaptic_change, 
            "duration": duration,
            "dreams_replayed": len(dream_contents)
        }

    def _apply_synaptic_scaling(self):
        """
        全ての重みを一定の割合で減少させる（乗算）。
        """
        if not isinstance(self.cortex_snn, torch.nn.Module):
            return

        with torch.no_grad():
            for param in self.cortex_snn.parameters():
                if param.requires_grad and param.dim() > 1: # 重み行列のみ（バイアスは除外など）
                    param.mul_(self.synaptic_scaling_factor)
        logger.info(f"   Refined synapses with scaling factor {self.synaptic_scaling_factor}.")

# 互換性のため import numpy が必要
import numpy as np
