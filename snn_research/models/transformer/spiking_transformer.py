# ファイルパス: snn_research/architectures/spiking_transformer_v2.py
# Title: Spiking Transformer v2 (SDSA統合版)
# Description:
# - 修正: forwardメソッド内で、get_total_spikes() を set_stateful(False) の前に移動。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional, Union, cast 
import math
import logging

# 必要なコアコンポーネントをインポート
from snn_research.core.base import BaseModel, SNNLayerNorm
from snn_research.core.neurons import AdaptiveLIFNeuron
# core.attention から SDSA をインポート
from snn_research.core.attention import SpikeDrivenSelfAttention
from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- PatchEmbedding クラス (変更なし) ---
class PatchEmbedding(nn.Module):
    """ 画像をパッチに分割し、線形射影する (ViTの入力層) """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

# --- SDSAEncoderLayer クラス ---
class SDSAEncoderLayer(sj_base.MemoryModule):
    """
    SDSAを使用したTransformerエンコーダーレイヤー。
    """
    input_spike_converter: AdaptiveLIFNeuron
    neuron_ff: AdaptiveLIFNeuron
    neuron_ff2: AdaptiveLIFNeuron
    sdsa: SpikeDrivenSelfAttention
    linear1: nn.Linear
    linear2: nn.Linear

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, time_steps: int, neuron_config: dict):
        super().__init__()
        # SDSAの初期化
        self.sdsa = SpikeDrivenSelfAttention(d_model, nhead, time_steps, neuron_config)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        
        lif_params = {k: v for k, v in neuron_config.items() if k in ['tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']}
        self.neuron_ff = cast(AdaptiveLIFNeuron, AdaptiveLIFNeuron(features=dim_feedforward, **lif_params))

        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.neuron_ff2 = cast(AdaptiveLIFNeuron, AdaptiveLIFNeuron(features=d_model, **lif_params))

        self.norm1 = SNNLayerNorm(d_model)
        self.norm2 = SNNLayerNorm(d_model)

        # 入力をスパイク化するためのニューロン
        lif_input_params = lif_params.copy() 
        lif_input_params['base_threshold'] = lif_input_params.get('base_threshold', 0.5) 
        self.input_spike_converter = cast(AdaptiveLIFNeuron, AdaptiveLIFNeuron(features=d_model, **lif_input_params))

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        self.sdsa.set_stateful(stateful)
        self.neuron_ff.set_stateful(stateful)
        self.neuron_ff2.set_stateful(stateful)
        self.input_spike_converter.set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        self.sdsa.reset()
        self.neuron_ff.reset()
        self.neuron_ff2.reset()
        self.input_spike_converter.reset()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # 1. SDSAによる自己注意
        # src はアナログ値を想定
        attn_output = self.sdsa(src)

        # 2. Residual Connection 1 + Norm 1
        # 入力をスパイク化して加算（あるいはアナログのまま加算するかは設計による）
        # ここでは入力をスパイク化して、SDSA出力（スパイクベースで計算されたアナログ値）と加算
        src_spiked, _ = self.input_spike_converter(src)
        
        # Add
        x = src_spiked + attn_output 
        x = torch.clamp(x, 0, 1) # スパイクライクに制限
        # Norm
        x_norm1 = self.norm1(x)

        # 3. Feedforward Network
        ff_spikes, _ = self.neuron_ff(self.linear1(x_norm1))
        ff_output_analog = self.linear2(ff_spikes)
        ff_output_spikes, _ = self.neuron_ff2(ff_output_analog)

        # 4. Residual Connection 2 + Norm 2
        x = x_norm1 + ff_output_spikes
        x = torch.clamp(x, 0, 1)
        x = self.norm2(x)

        return x

class SpikingTransformerV2(BaseModel):
    """
    SDSA Encoder Layer を使用した Spiking Transformer。
    ViT（画像）とテキストの両方に対応し、クロスモーダル注入をサポート。
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 nhead: int, 
                 num_encoder_layers: int, 
                 dim_feedforward: int, 
                 time_steps: int, 
                 neuron_config: Dict[str, Any],
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 **kwargs: Any):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embedding.num_patches
        
        self.pos_encoder_text = nn.Parameter(torch.zeros(1, 1024, d_model))
        self.pos_encoder_image = nn.Parameter(torch.zeros(1, num_patches, d_model))
        
        self.layers = nn.ModuleList([
            SDSAEncoderLayer(d_model, nhead, dim_feedforward, time_steps, neuron_config)
            for _ in range(num_encoder_layers)
        ])
        self.norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()
        logging.info(f"✅ SpikingTransformerV2 (SDSA, ViT, Cross-Modal) initialized.")

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                input_images: Optional[torch.Tensor] = None,
                # --- コンテキスト注入用引数 ---
                context_embeds: Optional[torch.Tensor] = None,
                return_spikes: bool = False, 
                output_hidden_states: bool = False, 
                **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids (Optional[torch.Tensor]): (B, SeqLen)
            input_images (Optional[torch.Tensor]): (B, C, H, W)
            context_embeds (Optional[torch.Tensor]): (B, CtxLen, D_model) - 外部から注入するコンテキスト
        """
        B: int
        N: int
        x: torch.Tensor
        device: torch.device
        
        SJ_F.reset_net(self)

        # 1. ベース入力の埋め込み
        if input_ids is not None:
            B, N = input_ids.shape
            device = input_ids.device
            x = self.token_embedding(input_ids)
            # 位置エンコーディング
            if N <= self.pos_encoder_text.shape[1]:
                 x = x + self.pos_encoder_text[:, :N, :]
            else:
                 x = x + self.pos_encoder_text[:, :self.pos_encoder_text.shape[1], :] # Truncate
        
        elif input_images is not None:
            device = input_images.device
            x = self.patch_embedding(input_images)
            B, N, _ = x.shape
            x = x + self.pos_encoder_image
        
        else:
            raise ValueError("Either input_ids or input_images must be provided.")

        # 2. Cross-Modal Injection (コンテキストの結合)
        # context_embeds がある場合、シーケンスの先頭に結合する (Prefix Tuning style)
        if context_embeds is not None:
            # context_embeds: (B, CtxLen, D_model)
            if context_embeds.shape[0] != B:
                 # バッチサイズが合わない場合の簡易対応 (ブロードキャスト)
                 if context_embeds.shape[0] == 1:
                      context_embeds = context_embeds.expand(B, -1, -1)
                 else:
                      logging.warning(f"Context batch size {context_embeds.shape[0]} != Input batch size {B}. Skipping injection.")
                      context_embeds = None

            if context_embeds is not None:
                logging.debug(f"Injecting context: {context_embeds.shape} into input: {x.shape}")
                x = torch.cat([context_embeds, x], dim=1) # (B, CtxLen + N, D_model)
                N = x.shape[1] # シーケンス長を更新

        # --- 時間ステップループ ---
        outputs_over_time = []

        # 各レイヤーをStatefulに設定
        for layer_module in self.layers:
             layer = cast(SDSAEncoderLayer, layer_module)
             layer.set_stateful(True)

        # レートコーディング入力: 毎ステップ、アナログ値 'x' を電流として入力
        # (本来はDVS入力などでx自体が時間変化するが、ここでは静的入力を仮定)
        for t in range(self.time_steps):
            x_step = x 

            for layer_module in self.layers:
                layer = cast(SDSAEncoderLayer, layer_module)
                x_step = layer(x_step) # スパイク出力

            outputs_over_time.append(x_step)

        x_final = torch.stack(outputs_over_time).mean(dim=0) # 時間平均 (アナログ)

        # --- 修正: リセット前にスパイクを集計 ---
        avg_spikes_val = 0.0
        if return_spikes:
            avg_spikes_val = self.get_total_spikes() / (B * N * self.time_steps)
        # ------------------------------------

        # Stateful解除
        for layer_module in self.layers:
             layer = cast(SDSAEncoderLayer, layer_module)
             layer.set_stateful(False)
        # --- ループ終了 ---

        x_final = self.norm(x_final)

        if output_hidden_states:
             output = x_final
        else:
            # 画像タスクかつコンテキストがない場合のプーリング
            if input_images is not None and context_embeds is None:
                pooled_output = x_final.mean(dim=1)
                output = self.output_projection(pooled_output)
            else:
                # テキストタスク または コンテキストあり
                output = self.output_projection(x_final)

        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return output, avg_spikes, mem