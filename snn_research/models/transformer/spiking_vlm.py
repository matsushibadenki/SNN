# ファイルパス: snn_research/models/transformer/spiking_vlm.py
# (Phase 3: Unified Multimodal Learning - Enhanced)
# Title: Spiking Unified Model (Formerly SpikingVLM)
# Description:
#   視覚、音声、触覚などの多感覚入力を UnifiedSensoryProjector 経由で統合し、
#   単一の Spiking Language Brain で処理する「単一学習エンジン」。

import torch
from typing import Dict, Any, Tuple
import logging

from snn_research.core.base import BaseModel
from snn_research.hybrid.multimodal_projector import UnifiedSensoryProjector
# 互換性のため古いクラス名もインポート可能にしておく

logger = logging.getLogger(__name__)


class SpikingUnifiedModel(BaseModel):
    """
    SNNベースの全感覚統合モデル。
    Structure: [Multi-Sensory Encoders] -> [Unified Projector] -> [Language/Reasoning Brain]
    """

    def __init__(
        self,
        vocab_size: int,
        language_config: Dict[str, Any],
        # {'vision': config, 'audio': config, ...}
        sensory_configs: Dict[str, Dict[str, Any]],
        projector_config: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        # --- Import Core ---
        try:
            from snn_research.core.snn_core import SNNCore
        except ImportError:
            raise ImportError("Failed to import SNNCore.")
        # -------------------

        logger.info(
            "🧠 SpikingUnifiedModel: Initializing Single Learning Engine...")

        # 1. Sensory Encoders (Brain Cortex Areas)
        # 各モダリティに対応するエンコーダを辞書として保持
        self.sensory_encoders = torch.nn.ModuleDict()
        modality_dims = {}

        for mod_name, config in sensory_configs.items():
            logger.info(f"   - Building Cortex Area: {mod_name}")
            # 出力次元の設定 (Projectorへの入力次元)
            output_dim = config.get("output_dim", 128)
            modality_dims[mod_name] = output_dim

            # SNNCoreを汎用エンコーダとして利用 (Configに応じてCNN/Linear等を切り替え)
            # vocab_size引数を出力次元として流用
            self.sensory_encoders[mod_name] = SNNCore(
                config=config, vocab_size=output_dim)

        # 2. Unified Sensory Projector (Thalamus/Bridge)
        logger.info("🔗 SpikingUnifiedModel: Building Unified Sensory Bridge...")
        self.projector = UnifiedSensoryProjector(
            language_dim=language_config.get("d_model", 256),
            modality_configs=modality_dims,
            use_bitnet=projector_config.get("use_bitnet", False)
        )

        # 3. Language/Reasoning Core (Prefrontal Cortex)
        logger.info("🗣️ SpikingUnifiedModel: Building Reasoning Core...")
        self.brain_core = SNNCore(
            config=language_config, vocab_size=vocab_size)

        self._init_weights()
        logger.info("✅ Single Learning Engine initialized.")

    def forward(
        self,
        input_ids: torch.Tensor,                    # (B, SeqLen) - テキスト思考/命令
        # {'vision': img, 'audio': wav, ...}
        sensory_inputs: Dict[str, torch.Tensor],
        return_spikes: bool = False,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unified Forward Pass
        """
        # 1. Encode All Senses (Parallel Processing)
        encoded_features = {}
        total_sensory_spikes = []

        for mod_name, encoder in self.sensory_encoders.items():
            if mod_name in sensory_inputs:
                raw_input = sensory_inputs[mod_name]
                outputs = encoder(raw_input)

                if isinstance(outputs, tuple):
                    feat = outputs[0]
                    spikes = outputs[1]
                else:
                    feat = outputs
                    spikes = torch.tensor(0.0, device=raw_input.device)

                encoded_features[mod_name] = feat
                if isinstance(spikes, torch.Tensor):
                    total_sensory_spikes.append(spikes.mean())

        # 2. Project to Unified Latent Space (Symbol Grounding)
        # ここで全ての感覚が「言語的な埋め込み」に変換・結合される
        context_embeds = self.projector(encoded_features)

        # 3. Reasoning / Language Generation with Context
        brain_outputs = self.brain_core(
            input_ids,
            context_embeds=context_embeds,
            return_spikes=True,
            **kwargs
        )

        logits = brain_outputs[0]
        brain_spikes = brain_outputs[1]
        mem = brain_outputs[2]

        # 全体のスパイク活動量を計算 (エネルギー効率モニタリング用)
        if total_sensory_spikes:
            avg_sensory_spike = torch.stack(total_sensory_spikes).mean()
        else:
            avg_sensory_spike = torch.tensor(0.0, device=logits.device)

        final_spike_rate = (avg_sensory_spike + brain_spikes.mean()) / 2.0

        return logits, final_spike_rate, mem

    def generate(
        self,
        input_ids: torch.Tensor,
        sensory_inputs: Dict[str, torch.Tensor],
        max_len: int = 20
    ) -> torch.Tensor:
        """
        多感覚入力に基づいた思考・応答生成
        """
        self.eval()
        with torch.no_grad():
            # 1. Encode
            encoded_features = {}
            for mod_name, encoder in self.sensory_encoders.items():
                if mod_name in sensory_inputs:
                    out = encoder(sensory_inputs[mod_name])
                    encoded_features[mod_name] = out[0] if isinstance(
                        out, tuple) else out

            # 2. Project
            context_embeds = self.projector(encoded_features)

            # 3. Generate Thought
            current_ids = input_ids
            for _ in range(max_len):
                outputs = self.brain_core(
                    current_ids, context_embeds=context_embeds)
                logits = outputs[0]
                next_token = torch.argmax(
                    logits[:, -1, :], dim=-1).unsqueeze(1)
                current_ids = torch.cat([current_ids, next_token], dim=1)

            return current_ids


# 互換性のためのエイリアス
SpikingVLM = SpikingUnifiedModel
