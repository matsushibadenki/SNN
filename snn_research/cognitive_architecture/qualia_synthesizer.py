# ファイルパス: snn_research/cognitive_architecture/qualia_synthesizer.py
# Title: Qualia Synthesizer (Consciousness Core)
# Description:
# - 感覚、感情、記憶を統合し、"主観的な質感"(Qualia)ベクトルを生成する。
# - Phase 6.2: Global Workspace Theory (GWT) のための内部表現生成器。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class QualiaSynthesizer(nn.Module):
    """
    クオリア生成モジュール。
    マルチモーダルな入力を統合し、意識のストリームに載せるための
    「現象的統合情報 (Phenomenal Integrated Information)」を生成する。
    """
    def __init__(self, sensory_dim: int = 256, emotion_dim: int = 64, memory_dim: int = 256, qualia_dim: int = 512):
        super().__init__()
        
        # 各モダリティのプロジェクション
        self.sensory_proj = nn.Linear(sensory_dim, qualia_dim)
        self.emotion_proj = nn.Linear(emotion_dim, qualia_dim)
        self.memory_proj = nn.Linear(memory_dim, qualia_dim)
        
        # クロスモーダル統合 (Attention Mechanism)
        self.integration_attention = nn.MultiheadAttention(embed_dim=qualia_dim, num_heads=4, batch_first=True)
        
        # クオリアバインディング層 (非線形統合)
        self.binding_layer = nn.Sequential(
            nn.Linear(qualia_dim, qualia_dim),
            nn.LayerNorm(qualia_dim),
            nn.GELU(),
            nn.Linear(qualia_dim, qualia_dim),
            nn.Tanh() # 範囲を制限
        )
        
        logger.info("✨ Qualia Synthesizer initialized.")

    def synthesize(self, 
                  sensory_input: Optional[torch.Tensor] = None, 
                  emotional_state: Optional[torch.Tensor] = None, 
                  memory_context: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        主観的体験（クオリア）を合成する。
        """
        batch_size = 1
        device = self.sensory_proj.weight.device
        
        # 1. 入力の正規化とエンベディング
        tokens = []
        
        if sensory_input is not None:
            s_emb = self.sensory_proj(sensory_input.to(device)).view(batch_size, 1, -1)
            tokens.append(s_emb)
            
        if emotional_state is not None:
            # 次元合わせ
            if emotional_state.numel() == 1:
                e_vec = torch.ones(1, self.emotion_proj.in_features, device=device) * emotional_state
            else:
                e_vec = emotional_state.to(device)
            e_emb = self.emotion_proj(e_vec).view(batch_size, 1, -1)
            tokens.append(e_emb)
            
        if memory_context is not None:
            m_emb = self.memory_proj(memory_context.to(device)).view(batch_size, 1, -1)
            tokens.append(m_emb)
            
        if not tokens:
            return {"qualia_vector": torch.zeros(1, 512, device=device), "complexity": 0.0}
            
        # [Batch, Seq, Dim]
        x = torch.cat(tokens, dim=1)
        
        # 2. 統合 (Self-Attention) - 異なる情報の関連付け
        attn_out, _ = self.integration_attention(x, x, x)
        
        # 3. バインディング - 一つの体験としての統合 (Pooling -> MLP)
        pooled = attn_out.mean(dim=1) # Global Average Pooling
        qualia_vector = self.binding_layer(pooled)
        
        # 統合情報量(Phi)の簡易推定 (ベクトルの分散を複雑性の指標とする)
        complexity = qualia_vector.std().item()
        
        return {
            "qualia_vector": qualia_vector, # これが「意識の内容」
            "components": len(tokens),
            "phi_proxy": complexity # 統合情報量の近似値
        }