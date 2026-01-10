# ファイルパス: snn_research/models/transformer/spiking_vlm.py
# 日本語タイトル: Spiking VLM (Alias & Captioning Support)
# 目的: Architecture RegistryやAgentからの呼び出しに対応するための修正。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

from snn_research.hybrid.multimodal_projector import MultimodalProjector

try:
    from snn_research.models.transformer.spikformer import Spikformer
except ImportError:
    Spikformer = nn.Linear  # type: ignore

try:
    from snn_research.models.cnn.spiking_cnn_model import SpikingCNN
except ImportError:
    SpikingCNN = nn.Linear  # type: ignore

try:
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
except ImportError:
    BitSpikeMamba = None  # type: ignore

logger = logging.getLogger(__name__)


class SpikingVLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        vision_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        language_config: Optional[Dict[str, Any]] = None,
        sensory_configs: Optional[Dict[str, Any]] = None,
        projector_config: Optional[Dict[str, Any]] = None,
        projection_dim: int = 512,
        use_bitnet: bool = True
    ):
        super().__init__()
        logger.info("👁️🗨️ Initializing SpikingVLM v2.2...")

        t_conf = text_config if text_config else (
            language_config if language_config else {})
        v_conf = vision_config if vision_config else {}

        if not v_conf and sensory_configs and 'vision' in sensory_configs:
            val = sensory_configs['vision']
            if isinstance(val, dict):
                v_conf = val
            else:
                v_conf = {'hidden_dim': val}

        self.vision_hidden_dim = v_conf.get("hidden_dim", 384)
        time_steps = v_conf.get("time_steps", 16)
        neuron_conf = v_conf.get("neuron", {"type": "lif"})
        self.vision_type = v_conf.get("type", "cnn")

        self.vision_encoder: nn.Module

        if self.vision_type == "bit_spike_mamba":
            if BitSpikeMamba is None:
                raise ImportError("BitSpikeMamba not found.")
            logger.info("   -> Using BitSpikeMamba")
            self.vision_encoder = BitSpikeMamba(
                vocab_size=0,
                d_model=self.vision_hidden_dim,
                d_state=v_conf.get("d_state", 16),
                d_conv=v_conf.get("d_conv", 4),
                expand=v_conf.get("expand", 2),
                num_layers=v_conf.get("num_layers", 2),
                time_steps=time_steps,
                neuron_config=neuron_conf
            )
            self.vision_proj = nn.Linear(3 * 32 * 32, self.vision_hidden_dim)

        elif self.vision_type == "cnn":
            logger.info("   -> Using SpikingCNN")
            self.vision_encoder = SpikingCNN(
                vocab_size=self.vision_hidden_dim,
                time_steps=time_steps,
                neuron_config=neuron_conf,
                img_size=v_conf.get("img_size", 32)
            )
        else:
            logger.warning("   -> Using Linear Placeholder")
            self.vision_encoder = nn.Linear(3*32*32, self.vision_hidden_dim)

        self.text_hidden_dim = t_conf.get("d_model", 512)
        self.text_model = nn.Embedding(vocab_size, self.text_hidden_dim)
        self.text_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.text_hidden_dim, nhead=8, batch_first=True),
            num_layers=t_conf.get("num_layers", 4)
        )

        p_dim = projector_config.get(
            "projection_dim", projection_dim) if projector_config else projection_dim
        self.projector = MultimodalProjector(
            vision_dim=self.vision_hidden_dim,
            text_dim=self.text_hidden_dim,
            embed_dim=p_dim,
            use_bitnet=use_bitnet
        )
        self.lm_head = nn.Linear(p_dim, vocab_size)

    def forward(self, image_input: torch.Tensor, text_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Vision Processing
        if self.vision_type == "bit_spike_mamba":
            B = image_input.shape[0]
            flat_img = image_input.view(B, 1, -1)
            emb_img = self.vision_proj(flat_img)
            mamba_out = self.vision_encoder(emb_img)
            vision_feats = mamba_out[0] if isinstance(
                mamba_out, tuple) else mamba_out
            if vision_feats.dim() == 2:
                vision_feats = vision_feats.unsqueeze(1)
        elif isinstance(self.vision_encoder, nn.Linear):
            B = image_input.shape[0]
            vision_feats = self.vision_encoder(
                image_input.view(B, -1)).unsqueeze(1)
        else:
            vision_out = self.vision_encoder(image_input)
            vision_feats = vision_out[0].unsqueeze(1) if isinstance(
                vision_out, tuple) else vision_out.unsqueeze(1)

        # Text Processing
        text_emb = self.text_model(text_input)
        text_feats = self.text_blocks(text_emb)

        # Projection & Fusion
        proj_output = self.projector(vision_feats, text_features=text_feats)

        if isinstance(proj_output, dict):
            align_loss = self.projector.compute_alignment_loss(proj_output)
            fused = proj_output["fused_representation"]
        else:
            align_loss = torch.tensor(0.0, device=image_input.device)
            fused = proj_output

        logits = self.lm_head(fused)

        return {
            "logits": logits,
            "alignment_loss": align_loss,
            "fused_representation": fused
        }

    def generate_caption(self, image_input: torch.Tensor, max_len: int = 20) -> torch.Tensor:
        """
        簡易的なキャプション生成メソッド。
        EmbodiedVLMAgentからの呼び出しに対応。
        """
        # 実運用ではここにBeam Searchなどを実装する
        # ここではダミーとしてランダムなトークン列を返す
        B = image_input.shape[0]
        # ダミーID列 [B, max_len]
        return torch.randint(0, 100, (B, max_len), device=image_input.device)


# Alias for backward compatibility and Registry
SpikingUnifiedModel = SpikingVLM
