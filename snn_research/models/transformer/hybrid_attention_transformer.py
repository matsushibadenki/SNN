# ファイルパス: snn_research/architectures/hybrid_attention_transformer.py
# (更新: AdaptiveAttentionModule 統合・スパイク集計修正)
# Title: Hybrid Attention Spiking Transformer (Adaptive Version)
# Description:
# - 修正: forwardメソッド内で、get_total_spikes() を set_stateful(False) の前に移動。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, cast
import math
import logging

# 必要なコアコンポーネントをインポート
from snn_research.core.base import BaseModel, SNNLayerNorm
from snn_research.core.neurons import AdaptiveLIFNeuron as LIFNeuron
from snn_research.core.adaptive_attention_selector import AdaptiveAttentionModule

# spikingjellyのfunctionalをリセットに利用
from spikingjelly.activation_based import functional # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AdaptiveTransformerLayer(nn.Module):
    """
    AdaptiveAttentionModule を使用した Transformer エンコーダーレイヤー。
    FFN部分はスパイクニューロンを使用。
    """
    neuron_ff: LIFNeuron
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float,
                 time_steps: int, neuron_config: dict):
        super().__init__()
        self.self_attn = AdaptiveAttentionModule(d_model, nhead, time_steps, neuron_config, dropout=dropout)

        # Feedforward Network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        lif_params_ffn = {k: v for k, v in neuron_config.items() if k in ['tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']}
        self.neuron_ff = LIFNeuron(features=dim_feedforward, **lif_params_ffn)
        self.dropout_ff = nn.Dropout(dropout) # FFN内のドロップアウト
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = SNNLayerNorm(d_model)
        self.norm2 = SNNLayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) # Attention後のドロップアウト
        self.dropout2 = nn.Dropout(dropout) # FFN後のドロップアウト

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): 入力シーケンス (Batch, SeqLen, Dim)。
            src_mask (Optional[torch.Tensor]): 標準Attention用のアテンションマスク。
            src_key_padding_mask (Optional[torch.Tensor]): 標準Attention用のキーパディングマスク。
        """
        # 1. Adaptive Attention + Residual Connection 1
        x = src
        # LayerNormをAttentionの前に入れることが多い
        x_norm1 = self.norm1(x)
        attn_output = self.self_attn(query=x_norm1, key=x_norm1, value=x_norm1,
                                     key_padding_mask=src_key_padding_mask,
                                     attn_mask=src_mask)
        x = x + self.dropout1(attn_output) # Residual Connection

        # 2. Feedforward Network + Residual Connection 2
        x_norm2 = self.norm2(x)
        ff_spikes, _ = self.neuron_ff(self.linear1(x_norm2)) # FFN内部はスパイク
        ff_output = self.linear2(self.dropout_ff(ff_spikes))
        x = x + self.dropout2(ff_output) # Residual Connection

        return x


class HybridAttentionTransformer(BaseModel):
    """
    AdaptiveAttentionModule を組み込んだ Spiking Transformer。
    """
    layers: nn.ModuleList

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 time_steps: int = 16,
                 neuron_config: Optional[Dict[str, Any]] = None,
                 max_seq_len: int = 512,
                 **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        if neuron_config is None:
            neuron_config = {'type': 'lif', 'tau_mem': 10.0, 'base_threshold': 1.0}

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            AdaptiveTransformerLayer(d_model, nhead, dim_feedforward, dropout, time_steps, neuron_config)
            for _ in range(num_layers)
        ])

        self.final_norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()
        logging.info("✅ HybridAttentionTransformer (Adaptive Version) initialized.")
        logging.info("   Each layer now uses AdaptiveAttentionModule.")

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = input_ids.shape
        device = input_ids.device

        functional.reset_net(self)

        # Embedding + Positional Encoding
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        if N > self.pos_encoder.shape[1]:
            logging.warning(f"Input sequence length ({N}) exceeds max_seq_len ({self.pos_encoder.shape[1]}) for positional encoding.")
            pos_encoding = self.pos_encoder[:, :N, :] # Truncate or handle differently?
        else:
            pos_encoding = self.pos_encoder[:, :N, :]
        x = x + pos_encoding
        x = self.embed_dropout(x)

        # --- 時間ステップループ (FFN内やSDSAパスのため) ---
        outputs_over_time = []

        # 各レイヤーのLIFニューロン等を Stateful に設定
        for layer_module in self.layers:
             layer = cast(AdaptiveTransformerLayer, layer_module)
             if hasattr(layer.neuron_ff, 'set_stateful'):
                  layer.neuron_ff.set_stateful(True)
             attn_module = layer.self_attn
             if isinstance(attn_module, AdaptiveAttentionModule):
                 if hasattr(attn_module.sdsa_attn, 'lif_q') and hasattr(attn_module.sdsa_attn.lif_q, 'set_stateful'):
                      attn_module.sdsa_attn.lif_q.set_stateful(True)
                      attn_module.sdsa_attn.lif_k.set_stateful(True)
                      attn_module.sdsa_attn.lif_v.set_stateful(True)


        current_x = x # ループ内で更新されるテンソル
        for t in range(self.time_steps):
            x_step = current_x # 前のステップの出力を入力とする

            # 各層を適用
            for layer_module in self.layers:
                layer = cast(AdaptiveTransformerLayer, layer_module)
                x_step = layer(x_step)

            outputs_over_time.append(x_step)
            current_x = x_step # 次のステップの入力のために更新

        # 時間平均を取る
        x_final = torch.stack(outputs_over_time).mean(dim=0)

        # --- 修正: リセット前にスパイクを集計 ---
        avg_spikes_val = 0.0
        if return_spikes:
            avg_spikes_val = self.get_total_spikes() / (B * N * self.time_steps)
        # ------------------------------------

        # 各レイヤーのLIFニューロン等を Stateless に戻す
        for layer_module in self.layers:
             layer = cast(AdaptiveTransformerLayer, layer_module)
             if hasattr(layer.neuron_ff, 'set_stateful'):
                  layer.neuron_ff.set_stateful(False)
             attn_module = layer.self_attn
             if isinstance(attn_module, AdaptiveAttentionModule):
                 if hasattr(attn_module.sdsa_attn, 'lif_q') and hasattr(attn_module.sdsa_attn.lif_q, 'set_stateful'):
                      attn_module.sdsa_attn.lif_q.set_stateful(False)
                      attn_module.sdsa_attn.lif_k.set_stateful(False)
                      attn_module.sdsa_attn.lif_v.set_stateful(False)

        # --- 時間ループ終了 ---

        x_final = self.final_norm(x_final)

        if output_hidden_states:
            output = x_final
        else:
            output = self.output_projection(x_final)

        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return output, avg_spikes, mem