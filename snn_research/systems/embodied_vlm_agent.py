# ファイルパス: snn_research/systems/embodied_vlm_agent.py
# 日本語タイトル: Embodied VLM Agent (Robust Version)
# 目的・内容:
#   ROADMAP Phase 2 "Multimodal Integration" の完成形。
#   [Fix] VLMからの出力キー欠落に対する堅牢なフォールバック処理を追加。

import torch
import torch.nn as nn
from typing import Dict, Any

from snn_research.models.transformer.spiking_vlm import SpikingVLM
from snn_research.models.motor.spiking_motor_decoder import SpikingMotorDecoder
import logging

logger = logging.getLogger(__name__)

class EmbodiedVLMAgent(nn.Module):
    """
    身体性を持つ視覚言語エージェント。
    """
    
    def __init__(
        self,
        vlm_model: SpikingVLM,
        motor_config: Dict[str, Any]
    ):
        super().__init__()
        self.vlm = vlm_model
        
        # VLMの融合表現の次元を取得 (Projectorの出力次元)
        fusion_dim = self.vlm.projector.embed_dim
        
        self.motor_decoder = SpikingMotorDecoder(
            input_dim=fusion_dim,
            action_dim=motor_config.get("action_dim", 6),
            action_type=motor_config.get("action_type", "continuous"),
            hidden_dim=motor_config.get("hidden_dim", 128)
        )
        
        logger.info("🤖 Embodied VLM Agent initialized (Vision+Language+Motor).")

    def forward(
        self, 
        image_input: torch.Tensor, 
        text_input: torch.Tensor
    ) -> Dict[str, Any]:
        """
        学習・推論パス
        """
        # 1. Perception & Cognition (VLM Forward)
        vlm_out = self.vlm(image_input, text_input)
        
        # 2. Extract Fused Representation
        fused_context = vlm_out.get("fused_representation")
        
        if fused_context is None:
             # Fallback: 次元不整合を防ぐため vision_latents を使用
             # vision_latents: [B, D] -> unsqueeze -> [B, 1, D]
             if "vision_latents" in vlm_out and len(vlm_out["vision_latents"]) > 0:
                 fused_context = vlm_out["vision_latents"].unsqueeze(1)
             else:
                 # 最悪ケース: ゼロ埋め
                 device = vlm_out["logits"].device
                 B = vlm_out["logits"].shape[0]
                 fused_context = torch.zeros(B, 1, self.motor_decoder.input_dim, device=device)

        # 3. Action Generation (Motor Decoder)
        action_output = self.motor_decoder(fused_context)
        
        return {
            "logits": vlm_out["logits"],         # For Language Loss
            "action_pred": action_output,        # For Action Loss
            "alignment_loss": vlm_out["alignment_loss"],
            "fused_context": fused_context
        }

    @torch.no_grad()
    def act_and_speak(
        self, 
        image_input: torch.Tensor, 
        prompt_input: torch.Tensor,
        max_len: int = 20
    ) -> Dict[str, Any]:
        """
        推論モード: 画像とプロンプトから、行動と発話を生成する。
        """
        self.eval()
        
        # 1. Generate Caption / Response
        generated_ids = self.vlm.generate_caption(image_input, max_len=max_len)
        
        # 2. Generate Action based on context
        vlm_out = self.vlm(image_input, prompt_input)
        fused_context = vlm_out.get("fused_representation")
        
        # Fallback Logic (Same as forward)
        if fused_context is None:
             if "vision_latents" in vlm_out and len(vlm_out["vision_latents"]) > 0:
                 fused_context = vlm_out["vision_latents"].unsqueeze(1)
             else:
                 device = image_input.device
                 B = image_input.shape[0]
                 fused_context = torch.zeros(B, 1, self.motor_decoder.input_dim, device=device)
        
        action = self.motor_decoder(fused_context)
        
        return {
            "generated_tokens": generated_ids,
            "action": action
        }