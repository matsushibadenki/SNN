# ファイルパス: snn_research/agent/active_inference_agent.py
# Title: Deep Active Inference Agent (Neural SNN-based) - Phase 4 Corrected
# Description:
#   ROADMAP Phase 4「能動的推論」の中核実装 (修正版)。
#   
#   修正内容:
#   - Transition Model (遷移モデル) を追加実装。ダミー計算を廃止し、
#     (状態, 行動) -> 次の状態 をニューラルネットワークで予測するように変更。
#   - update_model メソッドを拡張し、観測モデル(VAE的再構成)と遷移モデルの両方を学習するロジックに変更。
#   - 前回の信念 (prev_belief) を保持し、時間発展的な学習を可能に。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np

# SNNモデルの基底クラスをインポート
from snn_research.core.base import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActiveInferenceAgent:
    """
    SNNを生成モデル（観測モデル）として使用し、別途遷移モデルを持つ深層能動推論エージェント。
    """
    def __init__(
        self,
        generative_model: BaseModel,
        num_actions: int,
        action_dim: int,
        observation_dim: int,
        hidden_dim: int,
        lr: float = 0.01,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Args:
            generative_model (BaseModel): 観測モデル p(o|s) として機能するSNN。
            num_actions (int): 離散的な行動の数。
            action_dim (int): 行動の埋め込み次元。
            observation_dim (int): 観測データの次元。
            hidden_dim (int): 隠れ状態(Belief)の次元。
            lr (float): 学習率。
            optimizer (Optional[Optimizer]): 生成モデル更新用のオプティマイザ。
        """
        self.model = generative_model
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # 行動の埋め込み (Action Embedding)
        self.action_embedding = nn.Embedding(num_actions, action_dim)
        
        # --- 追加: 遷移モデル (Transition Model) p(s_{t+1} | s_t, a_t) ---
        # 以前のダミー計算 (+ mean * 0.1) を廃止し、学習可能なMLPに置き換え
        self.transition_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 選好分布 (Preference)
        self.preference_dist = torch.ones(observation_dim) / observation_dim
        
        # 状態管理
        self.current_belief: Optional[torch.Tensor] = None
        self.prev_belief: Optional[torch.Tensor] = None # 学習用に1ステップ前の信念を保持
        
        # オプティマイザの初期化
        if optimizer:
            self.optimizer = optimizer
        else:
            # モデル全体（観測モデル、遷移モデル、行動埋め込み）を最適化
            params = (
                list(self.model.parameters()) + 
                list(self.action_embedding.parameters()) +
                list(self.transition_model.parameters())
            )
            self.optimizer = torch.optim.Adam(params, lr=lr)
        
        logger.info("🤖 Deep Active Inference Agent initialized with Learnable Transition Model.")

    def reset(self):
        """エージェントの状態をリセットする。"""
        self.current_belief = torch.zeros(1, self.hidden_dim)
        self.prev_belief = None
        logger.info("Agent belief reset.")

    def set_preference(self, target_obs_vector: torch.Tensor):
        if target_obs_vector.shape[-1] != self.observation_dim:
            raise ValueError(f"Target dimension mismatch. Expected {self.observation_dim}, got {target_obs_vector.shape[-1]}")
        self.preference_dist = F.softmax(target_obs_vector, dim=-1)
        logger.info("Preference set: Updated target observation distribution.")

    def set_ethical_preference(self, avoid_indices: List[int], penalty_strength: float = 10.0):
        with torch.no_grad():
            logits = torch.log(self.preference_dist + 1e-8)
            for idx in avoid_indices:
                if 0 <= idx < self.observation_dim:
                    logits[idx] -= penalty_strength
            self.preference_dist = F.softmax(logits, dim=-1)
            logger.info(f"🛡️ Ethical preferences applied. Avoid indices: {avoid_indices}")

    def infer_state(self, observation: torch.Tensor) -> torch.Tensor:
        """
        知覚 (Perception): 観測から現在の信念 q(s) を推定する。
        """
        # 前回の信念を保存（学習用）
        if self.current_belief is not None:
            self.prev_belief = self.current_belief.detach().clone()

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(observation)
            
            # モデル出力のパース
            if isinstance(outputs, tuple):
                # FEELSNNなどは (logits, spikes, mem) を返す
                # 膜電位(mem)またはスパイク(spikes)を内部状態として採用
                state_repr = outputs[2] 
                if state_repr.numel() == 1: 
                     state_repr = outputs[0]
            else:
                state_repr = outputs

            # 次元合わせ
            if state_repr.shape[-1] != self.hidden_dim:
                 if state_repr.shape[-1] > self.hidden_dim:
                     state_repr = state_repr[..., :self.hidden_dim]
                 else:
                     pad = torch.zeros(*state_repr.shape[:-1], self.hidden_dim - state_repr.shape[-1], device=state_repr.device)
                     state_repr = torch.cat([state_repr, pad], dim=-1)
            
            self.current_belief = state_repr
            
        return self.current_belief

    def select_action(self, time_horizon: int = 1) -> int:
        """
        行動選択: 期待自由エネルギー G を最小化する行動を選ぶ。
        学習済み遷移モデルを使用してシミュレーションを行う。
        """
        if self.current_belief is None:
             return int(np.random.randint(0, self.num_actions))

        G_values = []
        
        for a in range(self.num_actions):
            # 行動埋め込み
            action_idx = torch.tensor([a], device=self.current_belief.device)
            action_emb = self.action_embedding(action_idx) # (1, action_dim)
            
            # 1. 遷移予測: q(s_{t+1} | s_t, a_t)
            # 修正: ニューラルネットワークによる遷移予測
            transition_input = torch.cat([self.current_belief, action_emb], dim=-1)
            predicted_next_state = self.transition_model(transition_input)
            
            # 2. 観測予測: q(o_{t+1} | s_{t+1})
            # ここでは簡易的に状態ベクトルをロジットとして扱う
            # (本来は Decoder p(o|s) を通すべきだが、潜在空間での距離で近似)
            predicted_obs_logits = predicted_next_state 
            
            # 次元調整
            if predicted_obs_logits.shape[-1] != self.observation_dim:
                 if predicted_obs_logits.shape[-1] > self.observation_dim:
                     predicted_obs_logits = predicted_obs_logits[..., :self.observation_dim]
                 else:
                     pad = torch.zeros(*predicted_obs_logits.shape[:-1], self.observation_dim - predicted_obs_logits.shape[-1], device=predicted_obs_logits.device)
                     predicted_obs_logits = torch.cat([predicted_obs_logits, pad], dim=-1)

            predicted_obs_dist = F.softmax(predicted_obs_logits, dim=-1)
            
            # 3. 期待自由エネルギー G の計算
            # Risk: 選好分布とのKL乖離
            risk = F.kl_div(predicted_obs_dist.log(), self.preference_dist, reduction='batchmean')
            
            # Ambiguity: エントロピー（不確実性）
            ambiguity = -torch.sum(predicted_obs_dist * torch.log(predicted_obs_dist + 1e-8))
            
            G = risk + ambiguity
            G_values.append(G.item())

        selected_action_idx = int(np.argmin(G_values))
        logger.info(f"Action selected: {selected_action_idx} (Min G={G_values[selected_action_idx]:.4f})")
        return selected_action_idx

    def update_model(self, observation: torch.Tensor, action: int, reward: float = 0.0):
        """
        学習ステップ:
        1. 観測モデル (Observation Model): 入力観測の再構成誤差 (VAE/AutoEncoder Loss)
        2. 遷移モデル (Transition Model): 予測した次状態と、実際に推論された次状態の誤差
        """
        if self.prev_belief is None or self.current_belief is None:
            # 履歴が足りない場合は学習できない
            logger.debug("Skipping update: Insufficient belief history.")
            return

        self.model.train()
        self.transition_model.train()
        self.optimizer.zero_grad()

        # --- A. 観測モデルの学習 (Reconstruction Loss) ---
        outputs = self.model(observation)
        if isinstance(outputs, tuple):
            reconstructed_logits = outputs[0]
        else:
            reconstructed_logits = outputs

        # 次元調整 (Loss計算用)
        if reconstructed_logits.shape[-1] > self.observation_dim:
             reconstructed_logits = reconstructed_logits[..., :self.observation_dim]
        
        # Loss計算 (Obs vs Reconstructed)
        if observation.shape == reconstructed_logits.shape:
             obs_loss = F.mse_loss(reconstructed_logits, observation)
        elif observation.dim() == reconstructed_logits.dim():
             target_dist = F.softmax(observation, dim=-1)
             pred_log_dist = F.log_softmax(reconstructed_logits, dim=-1)
             obs_loss = F.kl_div(pred_log_dist, target_dist, reduction='batchmean')
        else:
             obs_loss = torch.tensor(0.0, device=observation.device, requires_grad=True)

        # --- B. 遷移モデルの学習 (Transition Consistency Loss) ---
        # s_t (prev) と a_t から予測した s_{t+1} が、
        # 実際の o_{t+1} から推論した s_{t+1} (current) に近いか？
        
        action_idx = torch.tensor([action], device=self.prev_belief.device)
        action_emb = self.action_embedding(action_idx)
        
        transition_input = torch.cat([self.prev_belief, action_emb], dim=-1) # (B, hidden + action)
        predicted_next_state = self.transition_model(transition_input)     # (B, hidden)
        
        # 現在の信念(観測から導かれたもの)をターゲットとする
        # detachしてターゲットを固定し、遷移モデル側を近づける
        target_state = self.current_belief.detach()
        transition_loss = F.mse_loss(predicted_next_state, target_state)

        # 総損失
        total_loss = obs_loss + transition_loss

        # --- 更新 ---
        total_loss.backward()
        self.optimizer.step()
        
        logger.info(f"Model Updated. Total Loss: {total_loss.item():.4f} (Obs: {obs_loss.item():.4f}, Trans: {transition_loss.item():.4f})")
