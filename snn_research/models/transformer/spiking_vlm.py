# ファイルパス: snn_research/models/transformer/spiking_vlm.py
# 日本語タイトル: Spiking Vision-Language Model (SNN-VLM) v2.1 with Mamba Fix
# 目的・内容:
#   [Fix] BitSpikeMambaの出力次元の取り扱いを修正し、Projectorとの互換性を確保。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

from snn_research.hybrid.multimodal_projector import MultimodalProjector

# Models import
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
    """
    SNNベースの視覚言語モデル。
    """

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

        logger.info("👁️🗨️ Initializing SpikingVLM v2.1 (Mamba Fix)...")

        # Normalize configs
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

        # --- 1. Vision Encoder Selection ---
        self.vision_type = v_conf.get("type", "cnn")

        if self.vision_type == "bit_spike_mamba":
            if BitSpikeMamba is None:
                raise ImportError("BitSpikeMamba not found. Check imports.")

            logger.info("   -> Using BitSpikeMamba as Vision Encoder 🐍")
            # Mamba Config
            self.vision_encoder = BitSpikeMamba(
                vocab_size=0,  # Not used for vision embedding
                d_model=self.vision_hidden_dim,
                d_state=v_conf.get("d_state", 16),
                d_conv=v_conf.get("d_conv", 4),
                expand=v_conf.get("expand", 2),
                num_layers=v_conf.get("num_layers", 2),
                time_steps=time_steps,
                neuron_config=neuron_conf
            )

            # Image Patch Projection
            self.vision_proj = nn.Linear(3 * 32 * 32, self.vision_hidden_dim)

        elif self.vision_type == "cnn":
            logger.info("   -> Using SpikingCNN as Vision Encoder 🧠")
            self.vision_encoder = SpikingCNN(
                vocab_size=self.vision_hidden_dim,
                time_steps=time_steps,
                neuron_config=neuron_conf,
                img_size=v_conf.get("img_size", 32)
            )
        else:
            logger.warning("   -> Using Linear Placeholder for Vision")
            self.vision_encoder = nn.Linear(3*32*32, self.vision_hidden_dim)

        # --- 2. Text Model ---
        self.text_hidden_dim = t_conf.get("d_model", 512)
        self.text_model = nn.Embedding(vocab_size, self.text_hidden_dim)

        self.text_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.text_hidden_dim,
                nhead=8,
                batch_first=True
            ),
            num_layers=t_conf.get("num_layers", 4)
        )

        # --- 3. Multimodal Projector ---
        p_dim = projection_dim
        if projector_config:
            p_dim = projector_config.get("projection_dim", projection_dim)

        self.projector = MultimodalProjector(
            vision_dim=self.vision_hidden_dim,
            text_dim=self.text_hidden_dim,
            embed_dim=p_dim,
            use_bitnet=use_bitnet
        )

        self.lm_head = nn.Linear(p_dim, vocab_size)

    def forward(
        self,
        image_input: torch.Tensor,
        text_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        # A. Vision Encoding
        if self.vision_type == "bit_spike_mamba":
            B = image_input.shape[0]
            # [B, C, H, W] -> [B, 1, C*H*W]
            flat_img = image_input.view(B, 1, -1)
            # Project -> [B, 1, Vision_Dim]
            emb_img = self.vision_proj(flat_img)

            # Mamba Forward
            mamba_out = self.vision_encoder(emb_img)

            if isinstance(mamba_out, tuple):
                # (logits, spikes, mem)
                vision_feats = mamba_out[0]
            else:
                vision_feats = mamba_out

            # Ensure proper shape [B, T, D]
            # Mamba output might be [B, L, D]
            # If Time Loop in Mamba is 1 (for efficiency), L=1.
            # Projector expects [B, T, D]
            if vision_feats.dim() == 2:
                vision_feats = vision_feats.unsqueeze(1)

            # [Critial Check] Ensure last dimension matches vision_hidden_dim
            if vision_feats.shape[-1] != self.vision_hidden_dim:
                # This might happen if Mamba config overrides d_model internally or returns something else
                logger.warning(
                    f"Shape mismatch: {vision_feats.shape}, expected {self.vision_hidden_dim}")

        elif isinstance(self.vision_encoder, nn.Linear):
            B = image_input.shape[0]
            flat_img = image_input.view(B, -1)
            vision_feats = self.vision_encoder(flat_img).unsqueeze(1)
        else:
            # SpikingCNN
            vision_out = self.vision_encoder(image_input)
            if isinstance(vision_out, tuple):
                vision_feats = vision_out[0].unsqueeze(1)
            else:
                vision_feats = vision_out
                if vision_feats.dim() == 2:
                    vision_feats = vision_feats.unsqueeze(1)

        # B. Text Encoding
        text_emb = self.text_model(text_input)
        text_feats = self.text_blocks(text_emb)

        # C. Projection & Fusion
        # Calls Mode A of MultimodalProjector
        # Vision Feats: [B, 1, D_v]
        # Text Feats: [B, T_t, D_t]
        proj_output = self.projector(vision_feats, text_features=text_feats)

        # D. Output handling
        if isinstance(proj_output, dict):
            align_loss = self.projector.compute_alignment_loss(proj_output)
            fused = proj_output["fused_representation"]
            vision_latents = proj_output["vision_pooled"]
            text_latents = proj_output["text_pooled"]
        else:
            align_loss = torch.tensor(0.0, device=image_input.device)
            fused = proj_output
            vision_latents = torch.tensor([])
            text_latents = torch.tensor([])

        logits = self.lm_head(fused)

        return {
            "logits": logits,
            "alignment_loss": align_loss,
            "vision_latents": vision_latents,
            "text_latents": text_latents,
            "fused_representation": fused
        }

    @torch.no_grad()
    def generate_caption(self, image_input: torch.Tensor, max_len: int = 20) -> torch.Tensor:
        self.eval()
        B = image_input.shape[0]
        device = image_input.device

        curr_tokens = torch.tensor([[101]] * B, device=device)

        # Vision Encode (Pre-calculate)
        if self.vision_type == "bit_spike_mamba":
            B = image_input.shape[0]
            flat_img = image_input.view(B, 1, -1)
            emb_img = self.vision_proj(flat_img)
            mamba_out = self.vision_encoder(emb_img)
            if isinstance(mamba_out, tuple):
                vision_feats = mamba_out[0]
            else:
                vision_feats = mamba_out
            if vision_feats.dim() == 2:
                vision_feats = vision_feats.unsqueeze(1)

        elif isinstance(self.vision_encoder, nn.Linear):
            vision_feats = self.vision_encoder(
                image_input.view(B, -1)).unsqueeze(1)
        else:
            vision_out = self.vision_encoder(image_input)
            if isinstance(vision_out, tuple):
                vision_feats = vision_out[0].unsqueeze(1)
            else:
                vision_feats = vision_out.unsqueeze(1)

        for _ in range(max_len):
            text_emb = self.text_model(curr_tokens)
            text_feats = self.text_blocks(text_emb)

            proj_out = self.projector(vision_feats, text_features=text_feats)

            if isinstance(proj_out, dict):
                fused = proj_out["fused_representation"]
            else:
                fused = proj_out

            last_hidden = fused[:, -1, :]
            next_logit = self.lm_head(last_hidden)
            next_token = torch.argmax(next_logit, dim=-1, keepdim=True)

            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

            if (next_token == 102).all():
                break

        return curr_tokens


# Backward compatibility alias
SpikingUnifiedModel = SpikingVLM
