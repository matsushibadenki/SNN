# snn_research/models/experimental/concept_brain.py
# ファイルパス: snn_research/models/experimental/concept_brain.py
# 日本語タイトル: 概念統合脳モデル (Concept Integrated Brain)
# 説明: 視覚野（具体）と概念野（抽象）を統合し、トップダウンとボトムアップの情報を融合するモデル。

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from snn_research.models.bio.visual_cortex import VisualCortex

class ConceptBrain(nn.Module):
    """
    視覚情報（具体）と概念情報（抽象）を統合する脳モデル。
    """
    def __init__(self, num_classes: int = 10, embed_dim: int = 64):
        super().__init__()
        # 1. 視覚野 (Bottom-up Path)
        # 1ch (MNIST) -> Feature Map
        self.visual_cortex = VisualCortex(
            in_channels=1, 
            base_channels=16, 
            time_steps=4,
            neuron_params={'base_threshold': 1.0, 'v_reset': 0.0} 
        )
        
        # 視覚野の出力次元 (VisualCortexの仕様: base_channels * 8)
        visual_out_dim = 16 * 8 
        
        # 2. 統合層 (Integration / Association Area)
        # 視覚特徴を概念空間へ射影
        self.integrator = nn.Linear(visual_out_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
        # 3. 概念入力の射影 (Top-down Path)
        # NeuroSymbolicBridgeからの概念ベクトルを受け取る
        self.concept_proj = nn.Linear(embed_dim, embed_dim)
        
        # 4. 分類ヘッド (Task Output / Motor Cortex)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # 内部状態保持用（可視化・損失計算用）
        self.last_internal_state: Optional[torch.Tensor] = None

    def forward_sensory(self, x: torch.Tensor) -> torch.Tensor:
        """感覚入力（画像）のみを処理し、内部表現を返す"""
        # VisualCortexは (Batch, Time, Features) を返す
        features = self.visual_cortex(x)
        
        # 時間方向の平均を取り、安定した特徴量にする
        features_mean = features.mean(dim=1)
        
        # 統合層を通して概念空間と同じ次元へ
        return self.norm(self.integrator(features_mean))

    def forward_conceptual(self, concept_spikes: torch.Tensor) -> torch.Tensor:
        """概念入力（テキスト埋め込み）を処理し、トップダウン信号を返す"""
        return self.concept_proj(concept_spikes)

    def integrate(self, sensory_rep: torch.Tensor, conceptual_rep: Optional[torch.Tensor]) -> torch.Tensor:
        """
        感覚表現と概念表現を統合する。
        ここでの統合結果が「脳が認識している世界（内部状態）」となる。
        """
        # 概念ヒントがある場合は、感覚情報を概念で補正（バイアス）する
        # 森（概念）の情報で、木（具体）の見え方を調整するイメージ
        if conceptual_rep is not None:
            integrated = sensory_rep + 0.3 * conceptual_rep
        else:
            integrated = sensory_rep
            
        self.last_internal_state = integrated
        return self.classifier(integrated)

    def get_internal_state(self) -> torch.Tensor:
        """直前の推論時の内部状態（概念空間での座標）を返す"""
        if self.last_internal_state is None:
            raise RuntimeError("Internal state is empty. Run forward first.")
        return self.last_internal_state
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通常推論用（画像のみ、概念入力なし）"""
        sensory = self.forward_sensory(x)
        # 概念入力がない場合でも、integrateを経由してclassifierへ
        return self.integrate(sensory, None)