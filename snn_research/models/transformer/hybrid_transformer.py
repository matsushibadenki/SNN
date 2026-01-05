# ファイルパス: snn_research/architectures/hybrid_transformer.py
# (型ヒント修正)
# Title: Hybrid ANN-SNN Transformer (Gemma/GPT-like)
# Description:
# doc/SNN開発：Gemma3:GPT-4のSNN変換に関する技術的考察と戦略.md (戦略A/B) に基づき、
# Transformerのアーキテクチャをハイブリッド化します。
# - Attention, LayerNorm, Softmax はANN（アナログ）のまま保持します。
# - FFN (MLP) ブロックのみをSNN化（LIFニューロン）します。
# - ANNドメインとSNNドメインの境界にはアダプタ層を挿入します。
# mypy --strict 準拠。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, cast
import math
import logging

# プロジェクト内のモジュールをインポート
from snn_research.core.base import BaseModel
from snn_research.hybrid.adapter import AnalogToSpikes, SpikesToAnalog
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]

# 標準的なTransformerコンポーネント
from torch.nn import MultiheadAttention as StandardAttention
from torch.nn import LayerNorm as StandardLayerNorm
from torch.nn import Dropout

logger = logging.getLogger(__name__)

class HybridTransformerLayer(nn.Module):
    """
    ハイブリッドTransformerエンコーダ層。
    Attentionはアナログ、FFNはSNN。
    """
    norm1: StandardLayerNorm
    self_attn: StandardAttention
    dropout1: Dropout
    adapter_a2s: AnalogToSpikes
    snn_ffn_linear2: nn.Linear
    adapter_s2a: SpikesToAnalog
    norm2: StandardLayerNorm
    dropout2: Dropout
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        time_steps: int,
        neuron_config: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        
        # 1. Standard (Analog) Self-Attention Block
        self.norm1 = StandardLayerNorm(d_model)
        self.self_attn = StandardAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # 2. ANN -> SNN Adapter
        self.adapter_a2s = AnalogToSpikes(
            in_features=d_model,
            out_features=dim_feedforward, # FFNの入力次元に合わせる
            time_steps=time_steps,
            neuron_config=neuron_config
        )

        # 3. SNN Feedforward Block
        # FFNのSNN部分は、実際には AnalogToSpikes が Linear1 + Neuron を兼ねる
        # SNN (LIF) -> Linear の構成
        self.snn_ffn_linear2 = nn.Linear(dim_feedforward, d_model)

        # 4. SNN -> ANN Adapter
        self.adapter_s2a = SpikesToAnalog(
            in_features=d_model, # FFNの出力次元
            out_features=d_model,
            method="rate"
        )
        
        # 5. Residual Connection 2
        self.norm2 = StandardLayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self, 
        src_analog: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None, 
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src_analog (torch.Tensor): アナログ入力 (B, L, D_model)
            src_mask, src_key_padding_mask: 標準Attention用マスク
        
        Returns:
            torch.Tensor: アナログ出力 (B, L, D_model)
        """
        
        # --- 1. Analog Attention Block ---
        x: torch.Tensor = src_analog
        x_norm1: torch.Tensor = self.norm1(x)
        
        # StandardAttention は (attn_output, attn_weights) を返す
        attn_output_tuple: Tuple[torch.Tensor, Optional[torch.Tensor]] = self.self_attn(
            x_norm1, x_norm1, x_norm1, # Q, K, V
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
            need_weights=False # 通常は重み不要
        )
        attn_output: torch.Tensor = attn_output_tuple[0]
        
        x = x + self.dropout1(attn_output) # Residual Connection 1
        
        # --- 2. FFN Block (Hybrid) ---
        x_norm2: torch.Tensor = self.norm2(x) # (B, L, D_model)
        
        # 2a. ANN -> SNN (Adapter)
        # (B, L, D_model) -> (B, L, T, D_ff)
        ffn_spikes_time: torch.Tensor = self.adapter_a2s(x_norm2)
        
        # 2b. SNN Linear 2
        # (B, L, T, D_ff) -> (B*L*T, D_ff)
        B, L, T, D_ff = ffn_spikes_time.shape
        ffn_spikes_flat: torch.Tensor = ffn_spikes_time.reshape(B * L * T, D_ff)
        snn_ffn_output_flat: torch.Tensor = self.snn_ffn_linear2(ffn_spikes_flat)
        
        # (B*L*T, D_model) -> (B, L, T, D_model)
        snn_ffn_output_time: torch.Tensor = snn_ffn_output_flat.reshape(B, L, T, self.snn_ffn_linear2.out_features)
        
        # 2c. SNN -> ANN (Adapter)
        # (B, L, T, D_model) -> (B, L, D_model)
        ffn_output_analog: torch.Tensor = self.adapter_s2a(snn_ffn_output_time)
        
        x = x + self.dropout2(ffn_output_analog) # Residual Connection 2
        
        return x


class HybridSNNTransformer(BaseModel):
    """
    doc/SNN開発：Gemma3:GPT-4のSNN変換に関する技術的考察と戦略.md に基づく
    ハイブリッドANN-SNN Transformerモデル。
    """
    d_model: int
    embedding: nn.Embedding
    pos_encoder: nn.Parameter
    embed_dropout: Dropout
    layers: nn.ModuleList
    final_norm: StandardLayerNorm
    output_projection: nn.Linear

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        time_steps: int = 16, # SNN FFNが使用するT
        neuron_config: Optional[Dict[str, Any]] = None,
        max_seq_len: int = 512,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        if neuron_config is None:
            neuron_config = {'type': 'lif', 'tau_mem': 10.0, 'base_threshold': 1.0}

        self.d_model = d_model 

        # Analog (Standard) Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.embed_dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            HybridTransformerLayer(
                d_model, nhead, dim_feedforward, dropout, time_steps, neuron_config
            )
            for _ in range(num_layers)
        ])

        self.final_norm = StandardLayerNorm(d_model) # 最終NormもAnalog
        self.output_projection = nn.Linear(d_model, vocab_size) # 最終出力層

        self._init_weights() 
        logger.info("✅ Hybrid ANN-SNN Transformer (Strategy A/B) initialized.")

    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_spikes: bool = False, 
        output_hidden_states: bool = False, 
        return_full_hiddens: bool = False, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, N = input_ids.shape
        device: torch.device = input_ids.device
        
        # このモデルはSNN層の状態リセットを (SJ_F.reset_net) 呼ぶ必要がある
        SJ_F.reset_net(self)

        # 1. Analog Embedding
        x: torch.Tensor = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        pos_encoding: torch.Tensor
        if N > self.pos_encoder.shape[1]:
            logger.warning(f"Input sequence length ({N}) exceeds max_seq_len ({self.pos_encoder.shape[1]}) for positional encoding.")
            pos_encoding = self.pos_encoder[:, :N, :]
        else:
            pos_encoding = self.pos_encoder[:, :N, :]
        
        x = x + pos_encoding
        x = self.embed_dropout(x)

        # 2. Hybrid Encoder Layers
        # (このモデルでは内部で時間ループが回るため、外部ループは不要)
        for layer_module in self.layers:
            layer: HybridTransformerLayer = cast(HybridTransformerLayer, layer_module)
            # マスクの準備 (必要に応じてkwargsから取得)
            attn_mask: Optional[torch.Tensor] = kwargs.get('attn_mask')
            key_padding_mask: Optional[torch.Tensor] = kwargs.get('key_padding_mask')
            
            x = layer(
                src_analog=x, 
                src_mask=attn_mask, 
                src_key_padding_mask=key_padding_mask
            )
            
        # 3. Final Analog Output
        x = self.final_norm(x)

        output: torch.Tensor
        if output_hidden_states:
            output = x
        else:
            output = self.output_projection(x)
            
        # スパイク統計の計算
        avg_spikes_val: float = self.get_total_spikes() / (B * N * self.time_steps) if return_spikes and self.time_steps > 0 else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        
        # 最終膜電位 (このモデルでは直接的な意味はない)
        mem: torch.Tensor = torch.tensor(0.0, device=device)

        if return_full_hiddens:
             logger.warning("return_full_hiddens=True is not fully supported by HybridSNNTransformer, returning final hidden states instead.")
             return output.unsqueeze(2), avg_spikes, mem

        return output, avg_spikes, mem