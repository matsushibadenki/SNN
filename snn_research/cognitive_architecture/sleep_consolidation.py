# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# Title: Sleep Consolidation System (Generative Replay)
# Description:
#   ロードマップ Phase 5 実装。
#   睡眠サイクル中に、長期記憶（GraphRAG）から知識をランダムまたは重要度順に想起し、
#   「夢」としてSNNに入力することで、シナプス重みの固定化（Consolidation）と
#   構造化（Reorganization）を行う。

import torch
import logging
import random
import time
from typing import Dict, Any, List, Optional, Tuple

from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.core.networks.abstract_snn_network import AbstractSNNNetwork

logger = logging.getLogger(__name__)

class SleepConsolidator:
    """
    睡眠時の記憶固定化マネージャー。
    GraphRAGの知識をSNNのシナプス重みに焼き付ける（System 2 -> System 1）。
    """
    def __init__(
        self,
        rag_system: RAGSystem,
        cortex_snn: AbstractSNNNetwork,
        spike_encoder: SpikeEncoder,
        consolidation_epochs: int = 3,
        replay_batch_size: int = 4,
        replay_duration: int = 16
    ):
        """
        Args:
            rag_system: 知識源となるRAGシステム。
            cortex_snn: 学習対象のSNNモデル（通常はCorticalColumn）。
            spike_encoder: テキスト情報をスパイクに変換するエンコーダ。
            consolidation_epochs: 1回の睡眠サイクルでのリプレイ反復回数。
            replay_batch_size: リプレイ時のバッチサイズ。
            replay_duration: リプレイ信号の時間長（タイムステップ）。
        """
        self.rag_system = rag_system
        self.cortex_snn = cortex_snn
        self.spike_encoder = spike_encoder
        self.consolidation_epochs = consolidation_epochs
        self.replay_batch_size = replay_batch_size
        self.replay_duration = replay_duration
        
        logger.info("💤 SleepConsolidator initialized. Ready to dream.")

    def _generate_dreams(self, num_dreams: int = 10) -> List[Dict[str, Any]]:
        """
        GraphRAGから知識トリプルやドキュメントをサンプリングし、「夢」のコンテンツを生成する。
        最近追加された知識や、重要な概念（顕著性が高いもの）を優先するなどのロジックが可能。
        """
        dreams = []
        
        # 1. ナレッジグラフからのトリプル抽出
        if self.rag_system.knowledge_graph:
            try:
                edges = list(self.rag_system.knowledge_graph.edges(data=True))
                if edges:
                    # ランダムサンプリング（将来的には重要度サンプリングへ）
                    sample_size = min(len(edges), num_dreams)
                    sampled_edges = random.sample(edges, sample_size)
                    
                    for u, v, data in sampled_edges:
                        rel = data.get('relation', 'related_to')
                        # 自然言語風に変換
                        dream_text = f"{u} {rel} {v}."
                        dreams.append({
                            "content": dream_text,
                            "type": "knowledge_replay",
                            "source_triple": (u, rel, v)
                        })
            except Exception as e:
                logger.warning(f"Failed to sample from knowledge graph: {e}")

        # 2. ベクトルストアからのドキュメント抽出（不足分を補う）
        if len(dreams) < num_dreams and self.rag_system.vector_store:
            # ランダムなクエリで検索して多様性を確保
            random_queries = ["important", "concept", "structure", "memory", "snn"]
            query = random.choice(random_queries)
            try:
                docs = self.rag_system.search(query, k=num_dreams - len(dreams))
                for doc_text in docs:
                    dreams.append({
                        "content": doc_text,
                        "type": "episodic_replay"
                    })
            except Exception as e:
                logger.warning(f"Failed to search vector store: {e}")
        
        return dreams

    def perform_sleep_cycle(self) -> Dict[str, Any]:
        """
        睡眠サイクルを実行する。
        1. 夢（学習データ）の生成
        2. SNNへのリプレイ注入
        3. 可塑性による重み更新
        """
        logger.info("   🦄 Generating dreams from Knowledge Graph...")
        dreams = self._generate_dreams(num_dreams=self.replay_batch_size * 3)
        
        if not dreams:
            logger.warning("   ⚠️ No dreams generated. Knowledge base might be empty.")
            return {"synaptic_change": 0.0, "dreams_replayed": 0}

        total_synaptic_change = 0.0
        replayed_count = 0
        
        # ネットワークを学習モードに設定
        self.cortex_snn.train()
        
        # SNNの状態をリセット（睡眠開始）
        self.cortex_snn.reset_state()

        # リプレイ学習ループ
        for epoch in range(self.consolidation_epochs):
            random.shuffle(dreams)
            
            # バッチ処理
            for i in range(0, len(dreams), self.replay_batch_size):
                batch_dreams = dreams[i : i + self.replay_batch_size]
                if not batch_dreams:
                    continue
                
                # バッチ内のテキストをリスト化
                texts = [d["content"] for d in batch_dreams]
                
                # スパイクエンコーディング (Batch, Duration, Neurons)
                # SpikeEncoder.encode は単一入力を想定している場合があるため、バッチ処理用にラップ
                spike_inputs_list = []
                for text in texts:
                    # SensoryInfo形式
                    spikes = self.spike_encoder.encode({"content": text}, duration=self.replay_duration)
                    spike_inputs_list.append(spikes)
                
                # スタック: (Batch, Time, InputDim)
                try:
                    # Tensorの形状を合わせる必要がある
                    batch_spikes = torch.stack(spike_inputs_list, dim=0).to(next(self.cortex_snn.parameters()).device)
                except Exception as e:
                    logger.error(f"Error stacking spikes: {e}")
                    continue

                # 順伝播 (Forward)
                # ターゲットなしで実行（教師なし、または自己教師あり）
                # CorticalColumnの場合、内部で model_state に活動が記録される
                _ = self.cortex_snn(batch_spikes)
                
                # 学習ステップ (Plasticity Update)
                # ターゲットとして、入力そのもの（再構成）や、少し未来の予測などを与えることも可能
                # ここでは教師なし学習（STDP/CausalTrace）を想定し、targets=None
                # ただし、特定の「夢」に対して強い報酬信号（ドーパミン放出）を模倣してもよい
                metrics = self.cortex_snn.run_learning_step(inputs=batch_spikes, targets=None)
                
                # 変化量の集計
                update_mag = metrics.get("total_update_magnitude", 0.0)
                if isinstance(update_mag, torch.Tensor):
                    update_mag = update_mag.item()
                total_synaptic_change += update_mag
                
                replayed_count += len(batch_dreams)
                
                # ネットワーク状態のリセット (バッチ間)
                self.cortex_snn.reset_state()

        logger.info(f"   🧠 Sleep consolidation complete. Total Synaptic Change: {total_synaptic_change:.4f}")
        
        return {
            "synaptic_change": total_synaptic_change,
            "dreams_replayed": replayed_count,
            "epochs": self.consolidation_epochs
        }
