# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# 日本語タイトル: 睡眠時記憶固定化システム (GraphRAG to SNN)
# 機能説明: 
#   GraphRAGに蓄積された言語的・記号的な知識を、睡眠フェーズにおいて
#   SNN（大脳皮質モデル）への入力として再生し、STDP/BCM則を用いて
#   シナプス重みとして固定化（Consolidation）する機能。
#   これにより、「説明されたこと（宣言的記憶）」が「直感（手続き的記憶）」に変換される。

import torch
import logging
import random
from typing import List, Dict, Any, cast

# 既存モジュールのインポート
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.core.networks.bio_pc_network import BioPCNetwork
from snn_research.io.spike_encoder import SpikeEncoder

logger = logging.getLogger(__name__)

class SleepConsolidator:
    """
    睡眠中に記号的知識をニューラルネットワークの重みに変換するクラス。
    """
    def __init__(
        self, 
        rag_system: RAGSystem, 
        cortex_snn: BioPCNetwork, # 大脳皮質SNNモデル
        spike_encoder: SpikeEncoder,
        consolidation_epochs: int = 5
    ):
        self.rag_system = rag_system
        self.cortex_snn = cortex_snn
        self.spike_encoder = spike_encoder
        self.consolidation_epochs = consolidation_epochs

    def consolidate_knowledge(self, limit: int = 20) -> Dict[str, float]:
        """
        GraphRAGから知識を抽出し、SNNにリプレイ学習させる。
        
        Args:
            limit: 処理する知識エントリの最大数
            
        Returns:
            学習統計情報
        """
        logger.info("💤 Sleep Consolidation: Transferring GraphRAG knowledge to Synapses...")
        
        # 1. 重要な知識の抽出 (GraphRAGからランダムまたは重要度順に取得)
        # 注: RAGSystemに全ノード取得APIが必要だが、ここではnetworkxグラフに直接アクセスすると仮定
        if not self.rag_system.knowledge_graph:
            logger.warning("  - Knowledge graph is empty. Skipping consolidation.")
            return {}

        nodes = list(self.rag_system.knowledge_graph.nodes())
        selected_concepts = random.sample(nodes, min(len(nodes), limit))
        
        total_plasticity_change = 0.0
        
        # SNNを学習モードに設定
        self.cortex_snn.train()
        
        for epoch in range(self.consolidation_epochs):
            epoch_change = 0.0
            
            for concept in selected_concepts:
                # 知識の取得 (Subj - Pred - Obj)
                # 概念に関連するトリプルを取得してテキスト化
                triples = self.rag_system.get_subgraph_info(concept)
                if not triples:
                    continue
                    
                # 知識を一つの文脈として結合
                knowledge_text = " ".join(triples)
                
                # 2. スパイクエンコーディング
                # テキスト知識をスパイクパターンに変換
                # cortex_snn の入力次元に合わせる必要がある
                # BioPCNetworkは通常 (Batch, Dim) を受け取る
                input_dim = self.cortex_snn.layer_dims[0]
                
                # SpikeEncoderの既存メソッドを利用 (duration=time_steps)
                time_steps = self.cortex_snn.time_steps
                
                # テキストからスパイクを生成
                # encode メソッドは (Time, Neurons) を返す場合があるため調整
                # ここでは簡易的に dummy_input を生成するロジック (実際は encoder を使う)
                # spike_pattern = self.spike_encoder.encode({"content": knowledge_text}, duration=time_steps)
                
                # 【重要】記号接地されたエンコーディングが必要
                # ここでは概念ハッシュ等を使って決定論的なパターンを生成する簡易実装
                spike_pattern = self._generate_semantic_spikes(knowledge_text, input_dim, time_steps)
                spike_pattern = spike_pattern.to(next(self.cortex_snn.parameters()).device)

                # 3. SNNでの学習 (Forward & Plasticity Update)
                self.cortex_snn.reset_state()
                
                # 入力を与える (教師なし、あるいは自己教師あり)
                # BioPCNetworkは入力(x)がターゲット(target)にもなり得る (AutoEncoder的構成の場合)
                # ここでは入力を予測させる学習を行う
                
                # (Batch次元を追加)
                batch_input = spike_pattern.unsqueeze(0) # (1, Time, Dim) -> BioPCNetは (B, Dim) をT回入力?
                # BioPCNetwork.forward は (x, targets) を受け取る。
                # x: (B, Dim) (定常入力) または (B, T, Dim) (時系列)
                # 実装に合わせて (B, Dim) の平均レートを入力とする
                rate_input = batch_input.mean(dim=1)
                
                # Forward Pass
                _ = self.cortex_snn(rate_input, targets=rate_input) # 入力を再現するように学習
                
                # 学習則の適用 (Hebbian / STDP / Causal Trace)
                # run_learning_step は内部で登録されたルールを実行する
                metrics = self.cortex_snn.run_learning_step(inputs=rate_input, targets=rate_input)
                
                # 更新量の集計
                for k, v in metrics.items():
                    if "magnitude" in k:
                        epoch_change += v.item()
            
            total_plasticity_change += epoch_change
            logger.debug(f"  - Epoch {epoch+1} plasticity change: {epoch_change:.4f}")

        logger.info(f"✅ Consolidation complete. Total synaptic change: {total_plasticity_change:.4f}")
        return {"total_synaptic_change": total_plasticity_change}

    def _generate_semantic_spikes(self, text: str, dim: int, time_steps: int) -> torch.Tensor:
        """
        テキストから意味的に一貫したスパイクパターンを生成するヘルパー。
        (本来はSymbolGroundingやSpikeEncoderの機能だが、ここではデモ用に内包)
        """
        import hashlib
        import numpy as np
        
        # テキストのハッシュをシードにする
        seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        
        # レートコーディング的なパターンを生成
        # 意味的に近い単語が近いパターンになるわけではない簡易実装だが、
        # 「同じ知識」に対しては常に「同じスパイク」が生成されることが重要
        rate_vector = rng.rand(dim)
        
        spikes = []
        for _ in range(time_steps):
            s = (rng.rand(dim) < rate_vector).astype(np.float32)
            spikes.append(torch.from_numpy(s))
            
        return torch.stack(spikes)