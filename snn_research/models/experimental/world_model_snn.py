# ファイルパス: snn_research/models/experimental/world_model_snn.py
# Title: Spiking World Model (SWM) v1.2 - Tuple Unpacking Fix
# Description:
#   ROADMAP v17.5 "World Model" の先行実装。
#   修正: ニューロンがタプル(spikes, state)を返す仕様に対応し、nn.Sequentialを展開。

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, Union
import logging

from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.models.transformer.sformer import SFormerBlock

logger = logging.getLogger(__name__)

class SpikingWorldModel(nn.Module):
    """
    脳内シミュレーター。
    外部環境のダイナミクスを潜在空間(Latent Space)で学習・予測する。
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        action_dim: int = 10,
        d_state: int = 128,
        num_layers: int = 2,
        time_steps: int = 16,
        neuron_config: Optional[Dict[str, Any]] = None,
        input_dim: int = 128
    ):
        super().__init__()
        self.latent_dim = d_model
        self.time_steps = time_steps
        
        # 1. Encoder (Sensory -> Latent)
        if vocab_size > 0:
            self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.encoder_embedding = nn.Linear(input_dim, d_model) # type: ignore

        # Encoder Backend Layers (Sequentialだとタプル戻り値で死ぬため展開)
        self.enc_neuron1 = AdaptiveLIFNeuron(features=d_model)
        self.enc_linear = nn.Linear(d_model, d_model)
        self.enc_neuron2 = AdaptiveLIFNeuron(features=d_model)
        
        # 2. Predictor (Latent_t + Action_t -> Latent_t+1)
        self.action_encoder = nn.Linear(action_dim, d_model)
        
        self.predictor_blocks = nn.ModuleList([
            SFormerBlock(
                d_model=d_model, 
                nhead=4, 
                dim_feedforward=d_model*2,
                sf_threshold=2.0
            ) for _ in range(num_layers)
        ])
        
        self.transition_lif = AdaptiveLIFNeuron(features=d_model)
        
        # 3. Reward Predictor (Latent -> Reward)
        self.reward_head = nn.Linear(d_model, 1)
        
        logger.info(f"🌍 Spiking World Model initialized. Latent Dim: {d_model}, Action Dim: {action_dim}")

    def _apply_neuron(self, neuron_module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """ニューロン出力を安全に取り出すヘルパー"""
        out = neuron_module(x)
        if isinstance(out, tuple):
            return out[0] # (spikes, state) -> spikes
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """観測を潜在状態にエンコード"""
        # x: (Batch, InputDim) or (Batch, SeqLen)
        h = self.encoder_embedding(x)
        
        # Embeddingの場合、時間方向を平均するなどして状態ベクトル化 (簡易実装)
        if h.dim() == 3: 
            h = h.mean(dim=1)
            
        # Manual Forward for Backend
        h = self._apply_neuron(self.enc_neuron1, h)
        h = self.enc_linear(h)
        h = self._apply_neuron(self.enc_neuron2, h)
        
        return h

    def predict_next_step(self, current_latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1ステップ先の未来を予測する。
        Returns: next_latent, expected_reward
        """
        # Actionを埋め込み
        action_emb = self.action_encoder(action) # (Batch, Latent)
        
        # 状態と行動を結合
        x = current_latent + action_emb
        
        # Predictor (SFormer)
        # SFormerは(Batch, Seq, Dim)を期待するため次元追加
        x = x.unsqueeze(1) 
        for block in self.predictor_blocks:
            x = block(x)
        x = x.squeeze(1)
        
        # スパイク発火による非線形遷移
        next_latent = self._apply_neuron(self.transition_lif, x)
        
        # 報酬予測
        reward = self.reward_head(next_latent)
        
        return next_latent, reward

    def simulate_trajectory(
        self, 
        initial_state: torch.Tensor, 
        action_sequence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        [Mental Simulation]
        一連の行動（アクションプラン）を実行した場合の未来をシミュレーションする。
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