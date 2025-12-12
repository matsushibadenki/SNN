# ファイルパス: snn_research/models/experimental/world_model_snn.py
# Title: Spiking World Model (SWM) - "The Dreamer"
# Description:
#   Objective 7 & ROADMAP v17.5 実装。
#   環境のダイナミクス（状態遷移）を学習し、脳内シミュレーション（想像）を可能にする。
#   Dreamer (Hafner et al.) のアーキテクチャをSNN/SFormerで再構築したもの。
#   - Representation: 入力をスパイク潜在空間にエンコード。
#   - Transition: SFormerを用いて次の潜在状態を予測。
#   - Generation: 潜在状態から未来の観測を予測（夢を見る）。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List

from snn_research.core.base import BaseModel
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.models.transformer.sformer import SFormer
from snn_research.io.universal_encoder import UniversalSpikeEncoder

class SpikingWorldModel(BaseModel):
    """
    Spiking World Model (SWM)
    脳内で未来をシミュレーションするための生成モデル。
    """
    def __init__(
        self,
        vocab_size: int, # 観測空間の次元（離散トークンまたは量子化された特徴量）
        action_dim: int, # 行動空間の次元
        d_model: int = 256,
        d_state: int = 128, # 潜在状態の次元
        num_layers: int = 4,
        time_steps: int = 16,
        neuron_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.time_steps = time_steps
        
        # 1. Representation Model (Encoder)
        # 観測(x_t) -> 埋め込み
        self.observation_encoder = nn.Linear(vocab_size, d_model)
        self.action_encoder = nn.Linear(action_dim, d_model)
        
        # 2. Transition Model (Dynamics)
        # State(s_t) + Action(a_t) -> Next State(s_t+1)
        # 時系列予測に強いSFormerを採用
        self.transition_core = SFormer(
            vocab_size=1, # ダミー（内部でEmbeddingを使わないため）
            d_model=d_model,
            num_layers=num_layers,
            nhead=4,
            dropout=0.1,
            neuron_config=neuron_config
        )
        # SFormerのEmbedding層をバイパスするためのアダプタ
        self.state_projector = nn.Linear(d_model * 2, d_model) # [State, Action] -> d_model
        
        # 潜在状態のスパイク化（確率的発火による不確実性の表現）
        self.state_neuron = AdaptiveLIFNeuron(features=d_model, **(neuron_config or {}))
        
        # 3. Observation Model (Decoder)
        # State(s_t+1) -> Predicted Observation(x_t+1)
        self.observation_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(), # SNNならLIFだが、デコーダはアナログ値が必要な場合が多い
            nn.Linear(d_model, vocab_size)
        )
        
        # 4. Reward Model (Optional)
        # State(s_t+1) -> Predicted Reward(r_t+1)
        self.reward_predictor = nn.Linear(d_model, 1)

        self._init_weights()
        print(f"🌍 Spiking World Model initialized (State Dim: {d_state}). Ready to dream.")

    def forward(
        self, 
        obs: torch.Tensor, 
        action: torch.Tensor, 
        state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        1ステップの推論と学習用フォワードパス。
        
        Args:
            obs: (Batch, Vocab) - 現在の観測（One-hot or Features）
            action: (Batch, ActionDim) - 実行した行動
            state: (Batch, D_model) - 前回の潜在状態（なければ初期化）
        
        Returns:
            Dict: {
                "next_state": Tensor, 
                "pred_obs": Tensor, 
                "pred_reward": Tensor,
                "spikes": Tensor
            }
        """
        B = obs.shape[0]
        if state is None:
            state = torch.zeros(B, self.d_model, device=obs.device)
            
        # 1. Encode Observation & Action
        # (Batch, D_model)
        obs_emb = self.observation_encoder(obs) 
        act_emb = self.action_encoder(action)
        
        # 2. Transition (Prior)
        # 次の状態を予測: Transition(State_t-1, Action_t-1)
        # ここでは単純化のため、観測入力を事後分布(Posterior)として統合する構造にする
        # Input to Transition: [PrevState, Action]
        trans_input = torch.cat([state, act_emb], dim=-1)
        trans_input = self.state_projector(trans_input)
        
        # SFormerは本来シーケンス用だが、ここでは1ステップのRNN的に使用
        # (Batch, 1, D_model) にreshape
        trans_input_seq = trans_input.unsqueeze(1)
        
        # SFormer Forward (Embedding層をスキップするため、内部モジュールを直接呼ぶか、SFormerを改修する)
        # ここではSFormerの構造を活用しつつ、Embeddingを通さないハックを行う
        # -> SFormer.layers を直接使用
        x = trans_input_seq
        for layer in self.transition_core.layers:
            x = layer(x) # Maskなし
        x = self.transition_core.norm(x) # (Batch, 1, D_model)
        
        # 3. State Quantization (Spiking Latent)
        # 連続値をスパイク列に変換（状態の離散化・ボトルネック）
        analog_state = x.squeeze(1)
        next_state_spikes, _ = self.state_neuron(analog_state)
        # 勾配を通すためにStraight-Through Estimator的な扱い、あるいはアナログ値を保持
        # ここでは次ステップへの入力としてスパイクを使用（省エネ）
        next_state = next_state_spikes.float() # (Batch, D_model)
        
        # 4. Decode (Predict Future)
        pred_obs_logits = self.observation_decoder(next_state)
        pred_reward = self.reward_predictor(next_state)
        
        return {
            "next_state": next_state, # 次のステップの入力になる
            "pred_obs": pred_obs_logits,
            "pred_reward": pred_reward,
            "spikes": next_state_spikes,
            "analog_state": analog_state # 学習用（KL Divergence計算など）
        }

    def imagine_trajectory(
        self, 
        initial_state: torch.Tensor, 
        actions: torch.Tensor, 
        horizon: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        [Dreaming]
        初期状態から出発し、行動列に基づいて未来をシミュレーションする（観測なし）。
        プランニングや強化学習に使用。
        """
        trajectory = []
        current_state = initial_state
        
        for t in range(horizon):
            action = actions[:, t, :] # (Batch, ActionDim)
            
            # 観測がないため、Transition Modelのみで状態更新（Prior予測）
            # Encode Action
            act_emb = self.action_encoder(action)
            
            # Transition
            trans_input = torch.cat([current_state, act_emb], dim=-1)
            trans_input = self.state_projector(trans_input).unsqueeze(1)
            
            x = trans_input
            for layer in self.transition_core.layers:
                x = layer(x)
            x = self.transition_core.norm(x)
            
            analog_state = x.squeeze(1)
            state_spikes, _ = self.state_neuron(analog_state)
            current_state = state_spikes.float()
            
            # Decode
            pred_obs = self.observation_decoder(current_state)
            pred_reward = self.reward_predictor(current_state)
            
            trajectory.append({
                "state": current_state,
                "obs": pred_obs,
                "reward": pred_reward
            })
            
        return trajectory

    def reset_state(self):
        self.state_neuron.reset()
        # SFormer内部のニューロンもリセット
        if hasattr(self.transition_core, 'reset_state'):
            self.transition_core.reset_state()
        elif hasattr(self.transition_core, 'model') and hasattr(self.transition_core.model, 'reset_state'):
            self.transition_core.model.reset_state() # type: ignore