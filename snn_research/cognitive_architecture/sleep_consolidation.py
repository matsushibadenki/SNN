# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# Title: 睡眠時記憶固定化システム (GraphRAG to SNN Replay)
# Description:
#   ROADMAP Phase 5 "Neuro-Symbolic Evolution" の中核コンポーネント。
#   GraphRAGに蓄積された言語的・記号的な知識を、睡眠フェーズにおいて
#   SNN（大脳皮質モデル）への入力として再生し、STDP/BCM/Causal Trace則を用いて
#   シナプス重みとして固定化（Consolidation）する。

import torch
import logging
import random
from typing import List, Dict, Any, Optional
import networkx as nx

# 既存モジュールのインポート
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.core.networks.bio_pc_network import BioPCNetwork
# 循環参照回避のため型ヒントのみ
from snn_research.io.spike_encoder import SpikeEncoder

logger = logging.getLogger(__name__)

class SleepConsolidator:
    """
    睡眠中に記号的知識をニューラルネットワークの重みに変換するクラス。
    "Neuro-Symbolic Feedback Loop" の逆方向パス（記号 -> 神経）を担当する。
    """
    def __init__(
        self, 
        rag_system: RAGSystem, 
        cortex_snn: BioPCNetwork, # 学習対象のSNN
        spike_encoder: SpikeEncoder,
        consolidation_epochs: int = 3,
        replay_batch_size: int = 4
    ):
        self.rag_system = rag_system
        self.cortex_snn = cortex_snn
        self.spike_encoder = spike_encoder
        self.consolidation_epochs = consolidation_epochs
        self.replay_batch_size = replay_batch_size
        
        logger.info("💤 SleepConsolidator initialized. Ready to turn knowledge into intuition.")

    def _get_important_concepts(self, limit: int = 20) -> List[str]:
        """
        GraphRAGから固定化すべき重要な概念を抽出する。
        (PageRankや次数中心性などを用いて重要度を判定可能だが、現在はランダムサンプリング)
        """
        if not self.rag_system.knowledge_graph or self.rag_system.knowledge_graph.number_of_nodes() == 0:
            return []
            
        nodes = list(self.rag_system.knowledge_graph.nodes())
        # 簡易的にランダムサンプリング（将来的には活性度ベースに変更）
        selected = random.sample(nodes, min(len(nodes), limit))
        return selected

    def consolidate_knowledge(self) -> Dict[str, float]:
        """
        睡眠サイクルを実行し、知識をSNNに焼き付ける。
        """
        logger.info("💤 Sleep Phase: Replaying GraphRAG knowledge to Synapses...")
        
        selected_concepts = self._get_important_concepts()
        if not selected_concepts:
            logger.warning("  - No knowledge found to consolidate.")
            return {"total_synaptic_change": 0.0}

        total_plasticity_change = 0.0
        self.cortex_snn.train()
        
        # 学習ループ
        for epoch in range(self.consolidation_epochs):
            epoch_change = 0.0
            
            for concept in selected_concepts:
                # 1. 知識の再構成 (Symbol -> Text)
                # その概念に関連する知識トリプルを取得してテキスト化
                # ex: "SNN is energy_efficient."
                triples = self.rag_system.get_subgraph_info(concept)
                if not triples:
                    continue
                
                knowledge_text = " ".join(triples)
                
                # 2. スパイクエンコーディング (Text -> Spikes)
                # SpikeEncoderを使ってテキストをスパイク列に変換
                # SNNのtime_stepsに合わせる
                duration = self.cortex_snn.time_steps
                
                # input_dict形式で渡す
                spike_pattern = self.spike_encoder.encode(
                    {"content": knowledge_text}, 
                    duration=duration
                )
                
                # デバイス転送
                device = next(self.cortex_snn.parameters()).device
                spike_pattern = spike_pattern.to(device)
                
                # BioPCNetworkは (Batch, Dim) または (Batch, Time, Dim) を期待する
                # encodeは (T, N) を返す場合があるので調整
                if spike_pattern.dim() == 2: # (T, N)
                    # (1, T, N) -> (B, T, N)
                    model_input = spike_pattern.unsqueeze(0).repeat(self.replay_batch_size, 1, 1)
                else:
                    model_input = spike_pattern

                # 3. SNNでのリプレイ学習 (Forward & Plasticity Update)
                # 教師なし学習（Heobrian/STDP）または 自己教師あり学習（Predictive Coding）
                # 入力を「予測すべき対象」として与える
                
                self.cortex_snn.reset_state()
                
                # 入力をターゲットとしても使用 (Reconstruction)
                # BioPCNetworkのforward仕様に合わせて調整
                # forward(x) -> output
                _ = self.cortex_snn(model_input, targets=model_input) 
                
                # 学習則の適用
                metrics = self.cortex_snn.run_learning_step(inputs=model_input, targets=model_input)
                
                # 更新量の集計（ログ用）
                for k, v in metrics.items():
                    if "magnitude" in k and isinstance(v, torch.Tensor):
                        epoch_change += v.item()
                    elif "magnitude" in k and isinstance(v, float):
                        epoch_change += v

            total_plasticity_change += epoch_change
            logger.debug(f"  - Sleep Epoch {epoch+1}: Plasticity change magnitude = {epoch_change:.4f}")

        logger.info(f"✅ Consolidation complete. Knowledge integrated into synaptic weights. (Total change: {total_plasticity_change:.4f})")
        return {"total_synaptic_change": total_plasticity_change}
