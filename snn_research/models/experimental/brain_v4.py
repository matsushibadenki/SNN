# ファイルパス: snn_research/models/experimental/brain_v4.py
# 日本語タイトル: Brain v4.0 Synesthetic Architecture (Refactored)
# 目的: 視覚・聴覚・触覚・嗅覚を UnifiedSensoryProjector を通じて統合し、
#       単一の学習エンジンで処理する完全な共感覚アーキテクチャ。

import torch
import torch.nn as nn
from typing import Optional, Dict

from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.io.universal_encoder import UniversalSpikeEncoder
from snn_research.hybrid.multimodal_projector import UnifiedSensoryProjector


class SynestheticBrain(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 6,
        time_steps: int = 4,
        tactile_dim: int = 64,    # 触覚センサー入力の次元数
        olfactory_dim: int = 32,  # 嗅覚センサー入力の次元数
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.time_steps = time_steps
        self.d_model = d_model

        # 1. 汎用感覚野 (Universal Sensory Cortex)
        # 物理信号をスパイク列に変換
        self.encoder = UniversalSpikeEncoder(
            time_steps=time_steps,
            d_model=d_model,
            device=device
        ).to(device)

        # 2. 五感統合ブリッジ (Unified Sensory Bridge)
        # 全ての感覚モダリティの定義と射影をここで行う
        modality_configs = {
            'vision': 784,         # 画像特徴 (MNIST等)
            'audio': 64,           # 音響特徴 (MFCC等)
            'tactile': tactile_dim,
            'olfactory': olfactory_dim
        }

        self.sensory_bridge = UnifiedSensoryProjector(
            language_dim=d_model,
            modality_configs=modality_configs,
            use_bitnet=False
        ).to(device)

        # 3. 前頭葉・中枢エンジン (Central Executive)
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
                audio_input: Optional[torch.Tensor] = None,
                tactile_input: Optional[torch.Tensor] = None,
                olfactory_input: Optional[torch.Tensor] = None):
        """
        五感入力をスパイク化し、統合ブリッジを通してコンテキスト化する。
        """
        # Phase 1: 各感覚器でのスパイク符号化 (Encoding)
        sensory_spikes: Dict[str, torch.Tensor] = {}

        if image_input is not None:
            sensory_spikes['vision'] = self.encoder.encode(
                image_input, modality='image')

        if audio_input is not None:
            sensory_spikes['audio'] = self.encoder.encode(
                audio_input, modality='audio')

        if tactile_input is not None:
            sensory_spikes['tactile'] = self.encoder.encode(
                tactile_input, modality='tactile')

        if olfactory_input is not None:
            sensory_spikes['olfactory'] = self.encoder.encode(
                olfactory_input, modality='olfactory')

        # Phase 2: 感覚統合 (Fusion & Grounding)
        # UnifiedSensoryProjector が全てのスパイク入力を受け取り、統合コンテキストを返す
        sensory_context = self.sensory_bridge(
            sensory_spikes)  # [B, Total_Sensory_Seq, D]

        # Phase 3: 言語と思考 (Language & Thought)
        if text_input is not None:
            text_emb = self.core_brain.embedding(
                text_input)  # [B, Text_Seq, D]
            # 感覚コンテキストと思考列を結合
            combined_input = torch.cat([sensory_context, text_emb], dim=1)
        else:
            # テキスト入力がない場合でも、感覚情報だけで思考を開始できるようにする
            # ただし空の入力はエラーになるのでチェック
            if sensory_context.size(1) == 0:
                # 何も入力がない場合はダミー入力を入れるかエラーにする
                # ここではエラー回避のためダミーのstart token相当を入れる運用も考えられるが
                # ひとまずエラーとする
                raise ValueError("No input provided to SynestheticBrain")
            combined_input = sensory_context

        # Phase 4: 統合処理 (Unified Processing)
        logits, _, _ = self.core_brain(combined_input)

        return logits

    def generate(self, image_input: torch.Tensor, start_token_id: int, max_new_tokens: int = 20):
        self.eval()
        with torch.no_grad():
            # 視覚を知覚 -> 統合コンテキストへ
            vision_spikes = self.encoder.encode(image_input, modality='image')
            # 辞書で渡すのが UnifiedSensoryProjector の規約
            current_context = self.sensory_bridge({'vision': vision_spikes})

            generated_ids = []
            curr_input_ids = torch.tensor(
                [[start_token_id]], device=self.device)

            for _ in range(max_new_tokens):
                text_emb = self.core_brain.embedding(curr_input_ids)
                combined = torch.cat([current_context, text_emb], dim=1)
                logits, _, _ = self.core_brain(combined)

                next_token = torch.argmax(
                    logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(next_token.item())
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)

        return generated_ids
