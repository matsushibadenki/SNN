# ファイルパス: snn_research/models/experimental/brain_v4.py
# 日本語タイトル: Brain v4.0 Synesthetic Architecture [Fixed Device]
# 目的: 視覚・聴覚・言語を共通のスパイク信号として統合し、共感覚的な推論を可能にする。
# 修正: 全てのサブモジュールを確実に指定デバイスへ転送するよう修正。

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.io.universal_encoder import UniversalSpikeEncoder
from snn_research.hybrid.multimodal_projector import MultimodalProjector

class SynestheticBrain(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 6,
        time_steps: int = 4,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.time_steps = time_steps
        self.d_model = d_model

        # 1. 感覚野 (Sensory Cortex)
        # エンコーダーもnn.Moduleなのでデバイス転送が必要
        self.encoder = UniversalSpikeEncoder(
            time_steps=time_steps, 
            d_model=d_model, 
            device=device
        ).to(device)

        # 2. 連合野 (Association Cortex)
        # プロジェクターを定義し、即座にデバイスへ転送
        self.vision_projector = MultimodalProjector(
            visual_dim=784, # MNIST
            lang_dim=d_model,
            visual_time_steps=time_steps,
            lang_time_steps=time_steps
        ).to(device)
        
        self.audio_projector = MultimodalProjector(
            visual_dim=64, 
            lang_dim=d_model,
            visual_time_steps=time_steps,
            lang_time_steps=time_steps
        ).to(device)

        # 3. 前頭葉 (Prefrontal Cortex)
        self.core_brain = BitSpikeMamba(
            vocab_size=vocab_size,
            d_model=d_model,
            d_state=16, d_conv=4, expand=2,
            num_layers=num_layers,
            time_steps=time_steps,
            neuron_config={"type": "lif", "tau_mem": 2.0}
        ).to(device)

    def forward(self, 
                text_input: Optional[torch.Tensor] = None, 
                image_input: Optional[torch.Tensor] = None, 
                audio_input: Optional[torch.Tensor] = None):
        """
        あらゆる感覚入力を混合して思考する。
        """
        inputs_list = []

        # A. 視覚情報の処理
        if image_input is not None:
            # エンコーダーは内部でデバイス整合性をチェックするが、念のため
            vision_spikes = self.encoder.encode(image_input, modality='image', method='rate')
            vision_emb = self.vision_projector(vision_spikes) # [B, T, D]
            inputs_list.append(vision_emb)

        # B. 聴覚情報の処理
        if audio_input is not None:
            audio_spikes = self.encoder.encode(audio_input, modality='audio')
            audio_emb = self.audio_projector(audio_spikes)
            inputs_list.append(audio_emb)

        # C. テキスト情報の処理
        if text_input is not None:
            # テキスト入力をEmbedding層に通す
            text_emb = self.core_brain.embedding(text_input) # [B, L, D]
            inputs_list.append(text_emb)

        if not inputs_list:
            raise ValueError("No input provided to SynestheticBrain")

        # D. 感覚の統合 (Concatenate)
        combined_input = torch.cat(inputs_list, dim=1) # [B, Total_Len, D]

        # E. 思考
        logits, _, _ = self.core_brain(combined_input)
        
        return logits

    def generate(self, image_input: torch.Tensor, start_token_id: int, max_new_tokens: int = 20):
        """
        画像を見て、言葉(テキスト)を生成する
        """
        self.eval()
        with torch.no_grad():
            # 1. 画像を知覚
            vision_spikes = self.encoder.encode(image_input, modality='image')
            current_context = self.vision_projector(vision_spikes)
            
            # 2. 生成ループ
            generated_ids = []
            curr_input_ids = torch.tensor([[start_token_id]], device=self.device)
            
            for _ in range(max_new_tokens):
                text_emb = self.core_brain.embedding(curr_input_ids)
                combined = torch.cat([current_context, text_emb], dim=1)
                
                # SNNの状態をリセットしながら推論（簡易実装）
                # (高速化するにはKVキャッシュならぬMemキャッシュが必要だが今回は省略)
                logits, _, _ = self.core_brain(combined)
                
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_ids.append(next_token.item())
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
                
        return generated_ids