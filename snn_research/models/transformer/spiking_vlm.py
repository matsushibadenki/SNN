# ファイルパス: snn_research/models/transformer/spiking_vlm.py
# (Phase 3: Visual-Language Alignment - Bugfix)
# Title: Spiking Vision-Language Model (SpikingVLM)
# Description:
#   視覚エンコーダ (Vision Core) と言語モデル (Language Core) を
#   マルチモーダル・プロジェクターで接続した統合モデル。
#   修正: Vision Encoder の出力次元 (vocab_size) を projector の入力次元に合わせて動的に設定するように変更。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union, cast
import logging

from snn_research.core.base import BaseModel
from snn_research.hybrid.multimodal_projector import MultimodalProjector

logger = logging.getLogger(__name__)

class SpikingVLM(BaseModel):
    """
    SNNベースの視覚-言語統合モデル (VLM)。
    Structure: [Vision Encoder] -> [Projector] -> [Language Decoder]
    """
    def __init__(
        self,
        vocab_size: int,
        vision_config: Dict[str, Any],
        language_config: Dict[str, Any],
        projector_config: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        # --- 遅延インポートで循環参照を回避 ---
        try:
            from snn_research.core.snn_core import SNNCore
        except ImportError:
            raise ImportError("Failed to import SNNCore in SpikingVLM.")
        # ----------------------------------------
        
        # 1. Vision Encoder (e.g., SpikingCNN, SpikingViT)
        logger.info("👁️ SpikingVLM: Building Vision Encoder...")
        
        # 【修正】Projectorの入力次元に合わせてVision Encoderの出力次元(vocab_size)を設定
        # これにより、(B, 1000) ではなく (B, visual_dim) が出力され、型不一致エラーを防ぐ
        visual_dim = projector_config.get("visual_dim", 128)
        
        # Vision Encoderの出力クラス数として visual_dim を使用
        self.vision_encoder = SNNCore(config=vision_config, vocab_size=visual_dim)
        
        # 2. Multimodal Projector
        logger.info("🔗 SpikingVLM: Building Multimodal Projector...")
        self.projector = MultimodalProjector(
            visual_dim=visual_dim,
            lang_dim=language_config.get("d_model", 256),
            visual_time_steps=vision_config.get("time_steps", 16),
            lang_time_steps=language_config.get("time_steps", 16),
            use_bitnet=projector_config.get("use_bitnet", False)
        )
        
        # 3. Language Decoder (e.g., SpikingTransformer, RWKV)
        logger.info("🗣️ SpikingVLM: Building Language Decoder...")
        self.language_decoder = SNNCore(config=language_config, vocab_size=vocab_size)
        
        self._init_weights()
        logger.info("✅ SpikingVLM initialized successfully.")

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, SeqLen) - テキスト入力
        input_images: torch.Tensor,       # (B, C, H, W) - 画像入力
        return_spikes: bool = False,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for VLM.
        """
        # 1. Encode Images
        vision_outputs = self.vision_encoder(input_images)
        
        if isinstance(vision_outputs, tuple):
            visual_features = vision_outputs[0] # (B, VisualDim)
            vis_spikes = vision_outputs[1]
        else:
            visual_features = vision_outputs
            vis_spikes = torch.tensor(0.0, device=input_images.device)

        # 2. Project to Language Space
        context_embeds = self.projector(visual_features)
        
        # 3. Decode Text with Visual Context
        lang_outputs = self.language_decoder(
            input_ids, 
            context_embeds=context_embeds,
            return_spikes=True,
            **kwargs
        )
        
        logits = lang_outputs[0]
        lang_spikes = lang_outputs[1]
        mem = lang_outputs[2]
        
        # スパイク統計の統合 (Tensorかfloatかをチェックして計算)
        if isinstance(vis_spikes, torch.Tensor) and isinstance(lang_spikes, torch.Tensor):
            avg_spikes = (vis_spikes.mean() + lang_spikes.mean()) / 2.0
        elif isinstance(vis_spikes, torch.Tensor):
            avg_spikes = vis_spikes.mean()
        elif isinstance(lang_spikes, torch.Tensor):
            avg_spikes = lang_spikes.mean()
        else:
            avg_spikes = torch.tensor(0.0)
        
        return logits, avg_spikes, mem

    def generate(self, input_images: torch.Tensor, prompt_ids: torch.Tensor, max_len: int = 20) -> torch.Tensor:
        """
        画像を入力としてテキストを生成する (推論用)。
        """
        self.eval()
        with torch.no_grad():
            # 1. Vision Encoding
            vision_outputs = self.vision_encoder(input_images)
            visual_features = vision_outputs[0] if isinstance(vision_outputs, tuple) else vision_outputs
            
            # 2. Projection
            context_embeds = self.projector(visual_features)
            
            # 3. Autoregressive Generation
            current_ids = prompt_ids
            
            for _ in range(max_len):
                outputs = self.language_decoder(current_ids, context_embeds=context_embeds)
                logits = outputs[0]
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
            return current_ids