# ファイルパス: snn_research/systems/autonomous_learning_loop.py
# 日本語タイトル: Autonomous Learning Loop (Self-Supervised System) v1.1
# 目的・内容:
#   EmbodiedVLMAgent (Actor) と IntrinsicMotivator (Critic/Teacher) を結合。
#   [Fix] NameError: name 'F' is not defined を解消するためのインポート追加。

import torch
import torch.nn as nn
import torch.nn.functional as F  # Added
import torch.optim as optim
from typing import Dict, Any, List, Optional
import logging

from snn_research.systems.embodied_vlm_agent import EmbodiedVLMAgent
from snn_research.adaptive.intrinsic_motivator import IntrinsicMotivator

logger = logging.getLogger(__name__)

class AutonomousLearningLoop:
    """
    自律学習システム。
    Observation -> Action -> Observation' -> Reward(Internal) -> Update
    のサイクルを回す。
    """
    
    def __init__(
        self,
        agent: EmbodiedVLMAgent,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu"
    ):
        self.agent = agent.to(device)
        self.motivator = IntrinsicMotivator().to(device)
        self.optimizer = optimizer
        self.device = device
        
        # 予測ヘッド (Latent Predictor)
        # エージェントの「現在の状態+行動」から「次の状態」を予測する簡易モジュール
        fusion_dim = self.agent.motor_decoder.input_dim
        action_dim = self.agent.motor_decoder.action_dim
        
        self.world_predictor = nn.Sequential(
            nn.Linear(fusion_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, fusion_dim) # Predict next latent
        ).to(device)
        
        self.predictor_optimizer = optim.AdamW(self.world_predictor.parameters(), lr=1e-3)
        
        logger.info("🔄 Autonomous Learning Loop initialized.")

    def step(self, 
             current_image: torch.Tensor, 
             current_text: torch.Tensor,
             next_image: torch.Tensor # 環境からのフィードバック
             ) -> Dict[str, float]:
        """
        1ステップの自律学習サイクルを実行。
        """
        self.agent.train()
        self.world_predictor.train()
        
        # 1. Agent Perception & Action
        # Forward pass to get current latent and action
        agent_out = self.agent(current_image, current_text)
        
        z_t = agent_out["fused_context"] # [B, T, D] or [B, 1, D]
        action = agent_out["action_pred"]
        
        # 簡易化: 時間方向の平均または最後のステップを使用
        if z_t.dim() == 3:
            z_t_flat = z_t.mean(dim=1)
        else:
            z_t_flat = z_t
            
        # 2. World Prediction (What happens next?)
        # Predict z_{t+1} from z_t and action
        pred_input = torch.cat([z_t_flat, action], dim=-1)
        z_next_pred = self.world_predictor(pred_input)
        
        # 3. Observe Reality (Encode next image to get z_{t+1})
        # 教師データとしての「未来の自分」
        with torch.no_grad():
            # VLMを使って次の画像の潜在表現を取得 (Textは同じコンテキストを仮定)
            next_out = self.agent.vlm(next_image, current_text)
            z_next_actual = next_out["fused_representation"]
            
            # Fallback if None
            if z_next_actual is None:
                if "vision_latents" in next_out and len(next_out["vision_latents"]) > 0:
                    z_next_actual = next_out["vision_latents"].unsqueeze(1)
                else:
                    z_next_actual = torch.zeros_like(z_t) # dummy

            if z_next_actual.dim() == 3:
                z_next_actual_flat = z_next_actual.mean(dim=1)
            else:
                z_next_actual_flat = z_next_actual

        # 4. Compute Intrinsic Reward (Surprise)
        reward = self.motivator.compute_reward(z_next_pred, z_next_actual_flat)
        
        # 5. Losses
        # World Model Loss: 予測を現実に近づける
        wm_loss = F.mse_loss(z_next_pred, z_next_actual_flat)
        
        # Agent Loss: 
        # 本来は報酬最大化だが、デモ用としてwm_lossを逆伝播させる
        # 「予測しやすい潜在表現を獲得する」ようにEncoderを更新する。
        total_loss = wm_loss + agent_out["alignment_loss"] * 0.1
        
        # 6. Update
        self.optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()
        
        total_loss.backward()
        
        self.optimizer.step()
        self.predictor_optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "prediction_error": wm_loss.item(),
            "intrinsic_reward": reward.item(),
            "baseline": self.motivator.running_error_mean.item()
        }