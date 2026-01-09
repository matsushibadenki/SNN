# ファイルパス: snn_research/models/experimental/world_model_snn.py
# 日本語タイトル: Spiking World Model (Multimodal Edition)
# 目的: UnifiedSensoryProjectorを利用し、視覚だけでなく全感覚の未来状態を予測する世界モデル。
#       これにより、ロボットは「触った結果」や「音がどう変わるか」をシミュレーション可能になる。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from snn_research.core.base import BaseModel
from snn_research.core.snn_core import SNNCore
from snn_research.hybrid.multimodal_projector import UnifiedSensoryProjector
from snn_research.io.universal_encoder import UniversalSpikeEncoder


class SpikingWorldModel(BaseModel):
    """
    SNNベースのマルチモーダル世界モデル (JEPA / RSSM like architecture)

    [Observation] -> [Encoder] -> [Projector] -> [Latent State z_t]
                                      |
    [Action a_t] ---------------------+---> [Transition Model] -> [Predicted z_{t+1}]
                                      |
                                      +---> [Decoder] -> [Reconstructed Observation]
    """

    def __init__(
        self,
        vocab_size: int,  # Not strictly used in continuous world model, but kept for compatibility
        action_dim: int,
        d_model: int,
        d_state: int,
        num_layers: int,
        time_steps: int,
        sensory_configs: Dict[str, int],  # {'vision': 784, 'tactile': 64, ...}
        neuron_config: Dict[str, Any],
        **kwargs: Any
    ):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        self.action_dim = action_dim

        # 1. Perception (Encoder + Projector)
        self.encoder = UniversalSpikeEncoder(
            time_steps=time_steps, d_model=d_model)
        self.projector = UnifiedSensoryProjector(
            language_dim=d_model,
            modality_configs=sensory_configs,
            use_bitnet=kwargs.get("use_bitnet", False)
        )

        # 2. Action Encoder
        self.action_encoder = nn.Linear(action_dim, d_model)

        # 3. Transition Model (SNN / SSM Core)
        # 過去の状態と行動から、次の潜在状態を予測する
        self.transition_model = SNNCore(
            config={
                "d_model": d_model,
                "num_layers": num_layers,
                "time_steps": time_steps,
                "neuron": neuron_config,
                "architecture": "spiking_mamba"  # 高速な推論のためにMambaを採用
            },
            vocab_size=d_model  # 出力は次の潜在状態(d_model次元)
        )

        # 4. Decoder / Reward Predictor (Optional for reconstruction)
        # 潜在状態から各感覚を再構成するためのヘッド
        self.decoders = nn.ModuleDict()
        for mod, dim in sensory_configs.items():
            self.decoders[mod] = nn.Linear(d_model, dim)

        self._init_weights()

    def forward(
        self,
        sensory_inputs: Dict[str, torch.Tensor],
        actions: torch.Tensor,  # (B, T_seq, ActionDim)
        h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            sensory_inputs: 現在の観測 (各モダリティ)
            actions: 実行した行動
            h_prev: 前回の隠れ状態
        Returns:
            predicted_states: 予測された潜在状態列
            reconstructions: 再構成された観測
            h_next: 更新された隠れ状態
        """
        # --- 1. Encode Observation to Latent State z_t ---
        sensory_spikes = {}
        for mod, data in sensory_inputs.items():
            sensory_spikes[mod] = self.encoder.encode(data, modality=mod)

        # (B, T_seq, D)
        z_t = self.projector(sensory_spikes)

        # --- 2. Action Encoding ---
        # 行動を同じ次元に射影 (B, T_seq, D)
        a_t = self.action_encoder(actions)
        if a_t.size(1) != z_t.size(1):
            # 時間方向の長さが合わない場合は調整 (簡易実装)
            a_t = F.interpolate(a_t.transpose(
                1, 2), size=z_t.size(1)).transpose(1, 2)

        # --- 3. State Transition (Prediction) ---
        # 入力は「現在の状態 + 行動」
        # 本来はRNN的にステップごとに回すが、ここでは並列学習用にまとめて入力
        transition_input = z_t + a_t

        transition_out = self.transition_model(transition_input)

        if isinstance(transition_out, tuple):
            z_next_pred = transition_out[0]
            # spikes = transition_out[1]
            h_next = transition_out[2]
        else:
            z_next_pred = transition_out
            h_next = None

        # --- 4. Decode (Reconstruction) ---
        reconstructions = {}
        for mod, decoder in self.decoders.items():
            # 予測された潜在状態から観測を再構成
            reconstructions[mod] = decoder(z_next_pred)

        return z_next_pred, reconstructions, h_next

    def predict_next(
        self,
        current_sensory_inputs: Dict[str, torch.Tensor],
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        推論用: 現在の観測と行動から、次の瞬間の感覚を予測する (夢を見る機能)
        """
        self.eval()
        with torch.no_grad():
            z_pred, recons, _ = self(
                current_sensory_inputs, action.unsqueeze(1))

            # 再構成結果の最後のステップを返す
            next_senses = {k: v[:, -1, :] for k, v in recons.items()}
            return next_senses
