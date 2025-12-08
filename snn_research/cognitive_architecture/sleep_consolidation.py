# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# Title: Sleep Consolidation System (Generative Replay with Priority)
# Description:
#   ロードマップ Phase 5 実装の強化版。
#   睡眠サイクル中に、長期記憶（GraphRAG）から知識を想起し、「夢」としてSNNに入力する。
#   改善点:
#   - 優先度付きサンプリング (Prioritized Replay) を導入。
#     覚醒中に重要度（Salience）や予測誤差が高かった情報を優先的にリプレイする。
#   - NREM睡眠（構造化・汎化）とREM睡眠（結合・創造）のフェーズを模倣し、
#     記憶の固定化と創造的結合を使い分ける。

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
    優先度付きリプレイにより、重要な記憶を効率的に定着させる。
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
        
        logger.info("💤 SleepConsolidator initialized (Prioritized Replay enabled). Ready to dream.")

    def _calculate_priority(self, edge_data: Dict[str, Any]) -> float:
        """
        知識トリプルのリプレイ優先度を計算する。
        顕著性(salience)、感情価(valence)、または新規性に基づく。
        """
        # デフォルト優先度
        priority = 1.0
        
        # メタデータからの重み付け
        if 'salience' in edge_data:
            try:
                priority += float(edge_data['salience']) * 2.0
            except (ValueError, TypeError):
                pass
            
        if 'timestamp' in edge_data:
            # 新近性効果: 新しい記憶ほど優先されやすい（が、徐々に統合される）
            # ここでは簡易的に実装
            pass
            
        # 予測誤差や驚きが含まれている場合（metadataに記録されていると仮定）
        if 'surprise' in edge_data:
            try:
                priority += float(edge_data['surprise']) * 3.0
            except (ValueError, TypeError):
                pass
            
        return max(0.1, priority)

    def _generate_dreams(self, num_dreams: int = 10, phase: str = "NREM") -> List[Dict[str, Any]]:
        """
        GraphRAGから知識トリプルやドキュメントをサンプリングし、「夢」のコンテンツを生成する。
        
        Args:
            num_dreams: 生成する夢の数。
            phase: "NREM" (ノンレム睡眠: 事実の強化) または "REM" (レム睡眠: 遠隔連合・創造)。
        """
        dreams = []
        
        # 1. ナレッジグラフからのトリプル抽出
        if self.rag_system.knowledge_graph:
            try:
                edges = list(self.rag_system.knowledge_graph.edges(data=True))
                if edges:
                    # 優先度の計算
                    weighted_edges = []
                    for u, v, data in edges:
                        weight = self._calculate_priority(data)
                        weighted_edges.append((u, v, data, weight))
                    
                    # サンプリング戦略の切り替え
                    if phase == "NREM":
                        # NREM: 重要な事実（高優先度）を確実に固定化する
                        # 重み付きランダムサンプリング
                        weights = [w for _, _, _, w in weighted_edges]
                        sampled_indices = random.choices(
                            range(len(weighted_edges)), 
                            weights=weights, 
                            k=min(len(weighted_edges), num_dreams)
                        )
                        sampled_edges = [weighted_edges[i][:3] for i in sampled_indices]
                        
                    else: # REM
                        # REM: 関連性の薄いものやランダムな結合を試し、創造性を促す
                        # 一様ランダムサンプリングに近い、あるいは逆優先度
                        sampled_edges = random.sample(
                            [e[:3] for e in weighted_edges], 
                            min(len(weighted_edges), num_dreams)
                        )
                    
                    for u, v, data in sampled_edges:
                        rel = data.get('relation', 'related_to')
                        # 自然言語風に変換
                        dream_text = f"{u} {rel} {v}."
                        dreams.append({
                            "content": dream_text,
                            "type": "knowledge_replay",
                            "source_triple": (u, rel, v),
                            "phase": phase
                        })
            except Exception as e:
                logger.warning(f"Failed to sample from knowledge graph: {e}")

        # 2. ベクトルストアからのドキュメント抽出（不足分を補う）
        if len(dreams) < num_dreams and self.rag_system.vector_store:
            # ランダムなクエリで検索して多様性を確保
            random_queries = ["important", "concept", "structure", "memory", "snn", "future", "past"]
            query = random.choice(random_queries)
            try:
                # REM睡眠時はkを増やしてより広い範囲を探索
                search_k = num_dreams - len(dreams)
                if phase == "REM": search_k *= 2
                
                docs = self.rag_system.search(query, k=search_k)
                
                # 必要な数だけ追加
                for doc_text in docs[:num_dreams - len(dreams)]:
                    dreams.append({
                        "content": doc_text,
                        "type": "episodic_replay",
                        "phase": phase
                    })
            except Exception as e:
                logger.warning(f"Failed to search vector store: {e}")
        
        return dreams

    def perform_sleep_cycle(self) -> Dict[str, Any]:
        """
        睡眠サイクルを実行する。
        NREMフェーズ（事実固定）とREMフェーズ（統合・創造）を順に実行する。
        """
        total_synaptic_change = 0.0
        replayed_count = 0
        phases_executed = []

        # ネットワークを学習モードに設定
        self.cortex_snn.train()
        
        # 睡眠フェーズの定義
        sleep_schedule = ["NREM"] * 2 + ["REM"] * 1 # NREM重視のサイクル
        
        for phase_idx, phase in enumerate(sleep_schedule):
            logger.info(f"   💤 Sleep Phase {phase_idx+1}/{len(sleep_schedule)}: [{phase}] Generating dreams...")
            
            # フェーズに応じた夢の生成
            dreams = self._generate_dreams(
                num_dreams=self.replay_batch_size * 2, 
                phase=phase
            )
            
            if not dreams:
                continue

            # SNNの状態をリセット（フェーズ開始時）
            self.cortex_snn.reset_state()

            # リプレイ学習ループ
            # REMフェーズでは学習率を少し下げる（既存知識の破壊を防ぎつつ緩やかに統合するため）などの調整が可能
            # ここでは簡易的に共通処理とする
            
            random.shuffle(dreams)
            
            # バッチ処理
            for i in range(0, len(dreams), self.replay_batch_size):
                batch_dreams = dreams[i : i + self.replay_batch_size]
                if not batch_dreams:
                    continue
                
                # バッチ内のテキストをリスト化
                texts = [d["content"] for d in batch_dreams]
                
                # スパイクエンコーディング (Batch, Duration, Neurons)
                spike_inputs_list = []
                for text in texts:
                    spikes = self.spike_encoder.encode({"content": text}, duration=self.replay_duration)
                    spike_inputs_list.append(spikes)
                
                try:
                    # Tensorの形状を合わせる
                    device = next(self.cortex_snn.parameters()).device
                    batch_spikes = torch.stack(spike_inputs_list, dim=0).to(device)
                except Exception as e:
                    logger.error(f"Error stacking spikes: {e}")
                    continue

                # 順伝播 (Forward)
                # CorticalColumnの場合、内部で model_state に活動が記録される
                _ = self.cortex_snn(batch_spikes)
                
                # 学習ステップ (Plasticity Update)
                # 教師なし学習（STDP/CausalTrace）
                metrics = self.cortex_snn.run_learning_step(inputs=batch_spikes, targets=None)
                
                # 変化量の集計
                update_mag = metrics.get("total_update_magnitude", 0.0)
                if isinstance(update_mag, torch.Tensor):
                    update_mag = update_mag.item()
                total_synaptic_change += update_mag
                
                replayed_count += len(batch_dreams)
                
                # ネットワーク状態のリセット (バッチ間)
                self.cortex_snn.reset_state()
            
            phases_executed.append(phase)

        logger.info(f"   🧠 Sleep consolidation complete. Total Synaptic Change: {total_synaptic_change:.4f}")
        
        return {
            "synaptic_change": total_synaptic_change,
            "dreams_replayed": replayed_count,
            "phases": phases_executed
        }
