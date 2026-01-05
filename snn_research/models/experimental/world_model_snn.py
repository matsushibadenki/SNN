# ファイルパス: snn_research/models/experimental/world_model_snn.py
# Title: Spiking World Model (SWM) v1.3 - Optimization
# Description: 外部環境のダイナミクスを潜在空間で学習・予測する脳内シミュレーター。

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, Union
import logging

from snn_research.core.neurons import AdaptiveLIFNeuron

# SFormerBlockがない場合のフォールバック（単体テスト用）
try:
    from snn_research.models.transformer.sformer import SFormerBlock
except ImportError:
    # 簡易的なTransformerBlock
    class SFormerBlock(nn.Module): # type: ignore
        def __init__(self, d_model, **kwargs):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            self.ln = nn.LayerNorm(d_model)
        def forward(self, x):
            return self.ln(x + self.attn(x, x, x)[0])

logger = logging.getLogger(__name__)

class SpikingWorldModel(nn.Module):
    """
    脳内シミュレーター。
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        action_dim: int = 10,
        d_state: int = 128,
        num_layers: int = 2,
        time_steps: int = 16,
        input_dim: int = 128,
        neuron_config: Optional[Dict[str, Any]] = None, # [Fix] Added argument
        **kwargs: Any # [Fix] Added kwargs for flexibility
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Encoder (Sensory -> Latent)
        # vocab_size=0 の場合は連続値入力(input_dim)として扱う
        # [Fix] Type hint union
        self.encoder_embedding: Union[nn.Embedding, nn.Linear]
        if vocab_size > 0:
            self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.encoder_embedding = nn.Linear(input_dim, d_model)

        # ニューロン層 (状態保持)
        self.enc_neuron = AdaptiveLIFNeuron(features=d_model)
        
        # 2. Predictor (Latent_t + Action_t -> Latent_t+1)
        self.action_encoder = nn.Linear(action_dim, d_model)
        
        # 未来予測ブロック (Transformer-like)
        self.predictor_blocks = nn.ModuleList([
            SFormerBlock(
                d_model=d_model, 
                nhead=4, 
                dim_feedforward=d_model*2,
                sf_threshold=2.0
            ) for _ in range(num_layers)
        ])
        
        # 状態遷移の発火
        self.transition_lif = AdaptiveLIFNeuron(features=d_model)
        
        # 3. Reward Predictor (Latent -> Reward)
        # 報酬予測ヘッド (Critic)
        self.reward_head = nn.Linear(d_model, 1)
        
        logger.info(f"🌍 Spiking World Model initialized. Latent: {d_model}, Action: {action_dim}")

    def _apply_neuron(self, neuron_module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = neuron_module(x)
        if isinstance(out, tuple):
            return out[0]
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """観測を潜在状態にエンコード"""
        h = self.encoder_embedding(x)
        if h.dim() == 3: 
            h = h.mean(dim=1)
        h = self._apply_neuron(self.enc_neuron, h)
        return h

    def predict_next_step(self, current_latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """1ステップ先の予測"""
        # 行動の統合
        action_emb = self.action_encoder(action)
        x = current_latent + action_emb
        
        # 系列として処理するために次元追加 (B, 1, D)
        x = x.unsqueeze(1) 
        for block in self.predictor_blocks:
            x = block(x)
        x = x.squeeze(1)
        
        # 次の状態
        next_latent = self._apply_neuron(self.transition_lif, x)
        
        # 報酬予測
        reward = self.reward_head(next_latent)
        
        return next_latent, reward

    def simulate_trajectory(self, initial_state: torch.Tensor, action_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        [Mental Simulation] 行動計画のシミュレーション
        """
        B, Steps, _ = action_sequence.shape
        current_state = initial_state
        
        traj_states = []
        traj_rewards = []
        
        for t in range(Steps):
            action = action_sequence[:, t, :]
            next_state, reward = self.predict_next_step(current_state, action)
            
            traj_states.append(next_state)
            traj_rewards.append(reward)
            current_state = next_state
            
        return {
            "states": torch.stack(traj_states, dim=1),
            "rewards": torch.stack(traj_rewards, dim=1),
            "final_state": current_state
        }