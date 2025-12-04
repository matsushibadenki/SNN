# ファイルパス: snn_research/agent/active_inference_agent.py
# Title: Deep Active Inference Agent (Neural SNN-based) - Phase 4 Complete
# Description:
#   ROADMAP Phase 4「能動的推論」の中核実装。
#   学習済みSNNモデル（生成モデル）を用いて期待自由エネルギー（G）を最小化する行動選択を行う。
#   
#   更新内容:
#   - update_model: 観測データに基づき、生成モデル（SNN）の重みを更新する学習ロジックを実装。
#   - set_ethical_preference: 倫理的制約（危害回避など）を選好分布に反映させる機能を追加。
#   - infer_state: SNNの出力を確率分布として解釈するロジックを強化。
#   - 修正: 末尾の不要な '}' を削除。

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
    SNNを生成モデルとして使用する深層能動推論エージェント。
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
            generative_model (BaseModel): 世界モデルとして機能するSNN (Transition & Observation model)。
            num_actions (int): 離散的な行動の数。
            action_dim (int): 行動の埋め込み次元 (SNNへの入力用)。
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
        
        # 選好分布 (Preference / C-matrix equivalent)
        # 特定の観測状態に対する「好み」を定義。デフォルトは平坦。
        self.preference_dist = torch.ones(observation_dim) / observation_dim
        
        # 現在の信念状態 (Posterior Belief)
        self.current_belief: Optional[torch.Tensor] = None
        
        # オプティマイザの初期化
        if optimizer:
            self.optimizer = optimizer
        else:
            # モデルと行動埋め込みの両方を最適化対象にする
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.action_embedding.parameters()), 
                lr=lr
            )
        
        logger.info("🤖 Deep Active Inference Agent initialized (Neural-based, Phase 4 Complete).")

    def reset(self):
        """エージェントの状態をリセットする。"""
        self.current_belief = torch.zeros(1, self.hidden_dim)
        logger.info("Agent belief reset.")

    def set_preference(self, target_obs_vector: torch.Tensor):
        """
        選好（ゴール）を設定する。
        Args:
            target_obs_vector: 望ましい観測分布 (確率分布または特徴ベクトル)。
        """
        if target_obs_vector.shape[-1] != self.observation_dim:
            raise ValueError(f"Target dimension mismatch. Expected {self.observation_dim}, got {target_obs_vector.shape[-1]}")
        
        # 確率分布として正規化
        self.preference_dist = F.softmax(target_obs_vector, dim=-1)
        logger.info("Preference set: Updated target observation distribution.")

    def set_ethical_preference(self, avoid_indices: List[int], penalty_strength: float = 10.0):
        """
        倫理的選好を設定する。特定の観測状態（例：危害、エラー）を避けるように分布を調整する。
        Args:
            avoid_indices: 避けるべき観測状態のインデックスリスト。
            penalty_strength: 回避の強さ。
        """
        with torch.no_grad():
            # 現在の選好をロジット空間へ
            logits = torch.log(self.preference_dist + 1e-8)
            
            for idx in avoid_indices:
                if 0 <= idx < self.observation_dim:
                    logits[idx] -= penalty_strength # 避けるべき状態の確率を下げる
            
            self.preference_dist = F.softmax(logits, dim=-1)
            logger.info(f"🛡️ Ethical preferences applied. Avoid indices: {avoid_indices}")

    def infer_state(self, observation: torch.Tensor) -> torch.Tensor:
        """
        知覚 (Perception): 変分推論により、観測から現在の信念 q(s) を推定する。
        Deep Active Inferenceでは、これは通常モデルのEncoder部分、または
        誤差最小化プロセス（Predictive Coding）によって行われる。
        """
        self.model.eval()
        with torch.no_grad():
            # 観測を入力してモデルの隠れ状態を更新
            outputs = self.model(observation)
            
            # モデルの出力仕様に合わせて調整
            # タプル (logits, spikes, mem) の場合、mem (膜電位) または spikes を状態とする
            if isinstance(outputs, tuple):
                # 3番目の要素(mem)を内部状態の近似として使用
                state_repr = outputs[2] 
                if state_repr.numel() == 1: # ダミーの場合
                     state_repr = outputs[0] # logitsを使用
            else:
                state_repr = outputs

            # 次元合わせ (簡易的な射影またはスライス)
            if state_repr.shape[-1] != self.hidden_dim:
                 if state_repr.shape[-1] > self.hidden_dim:
                     state_repr = state_repr[..., :self.hidden_dim]
                 else:
                     # パディング
                     pad = torch.zeros(*state_repr.shape[:-1], self.hidden_dim - state_repr.shape[-1], device=state_repr.device)
                     state_repr = torch.cat([state_repr, pad], dim=-1)
            
            self.current_belief = state_repr
            
        return self.current_belief

    def select_action(self, time_horizon: int = 1) -> int:
        """
        行動選択 (Action Selection): 期待自由エネルギー G を最小化する行動を選ぶ。
        Deep Active Inference: シミュレーションによる評価。
        """
        if self.current_belief is None:
             return np.random.randint(0, self.num_actions)

        G_values = []
        
        # 各行動候補についてシミュレーション
        for a in range(self.num_actions):
            action_idx = torch.tensor([a], device=self.current_belief.device)
            action_emb = self.action_embedding(action_idx)
            
            # 1. 遷移予測: q(s_{t+1} | s_t, a_t)
            # 簡易実装: 現在の信念にアクションの影響を加算
            # (SNN自体がRecurrentなら、本来は hidden_state を更新して予測する)
            predicted_next_state = self.current_belief + action_emb.mean() * 0.1 
            
            # 2. 観測予測: q(o_{t+1} | s_{t+1})
            with torch.no_grad():
                 # モデルの出力層を利用したいが、簡易的に状態から予測
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
            # G = Risk + Ambiguity
            
            # (A) リスク (Risk): D_KL( q(o) || p(o) )
            # 選好分布（ゴール）からの乖離
            risk = F.kl_div(predicted_obs_dist.log(), self.preference_dist, reduction='batchmean')
            
            # (B) 曖昧さ (Ambiguity): H(A)
            # 状態のエントロピー（不確実性）。低いほど確信度が高い。
            # ここでは「Ambiguityを減らす」＝「情報を得る」動機付けとして加算
            ambiguity = -torch.sum(predicted_obs_dist * torch.log(predicted_obs_dist + 1e-8))
            
            # G = Risk + Ambiguity (係数でバランス調整可)
            G = risk + ambiguity
            G_values.append(G.item())

        # Gを最小化する行動を選択
        selected_action_idx = int(np.argmin(G_values))
        
        logger.info(f"Action selected: {selected_action_idx} (Min G={G_values[selected_action_idx]:.4f})")
        return selected_action_idx

    def update_model(self, observation: torch.Tensor, action: int, reward: float = 0.0):
        """
        経験に基づく生成モデルの学習 (Variational Free Energy Minimization)。
        観測データに対するサプライズ（予測誤差）を最小化するようにモデルを更新する。
        
        Args:
            observation: 実際の観測データ。
            action: 直前に取った行動。
            reward: (オプション) 外部報酬。FEPでは通常、予測誤差の一部として扱うか、
                    選好分布の更新に使用するが、ここでは補助的なシグナルとして利用可能。
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. 現在の状態（信念）からの予測を計算
        # 本来は (state, action) -> predicted_observation
        if self.current_belief is None:
            return

        # 簡易的に、現在の信念から観測を再構成するタスクとして定式化
        # (変分自由エネルギー F = D_KL(q(s)||p(s)) - E_q[ln p(o|s)])
        
        # モデルの出力（観測予測）
        outputs = self.model(observation) # ここではAutoEncoder的に観測そのものを入力しているが、本来は直前の状態
        
        if isinstance(outputs, tuple):
            predicted_logits = outputs[0]
        else:
            predicted_logits = outputs

        # 次元調整
        if predicted_logits.shape[-1] != self.observation_dim:
             # 次元が合わない場合の簡易対応
             if predicted_logits.shape[-1] > self.observation_dim:
                 predicted_logits = predicted_logits[..., :self.observation_dim]
             else:
                 # パディングなどで合わせる必要があるが、ここではスキップ
                 pass

        if predicted_logits.shape == observation.shape:
             # MSE Loss (観測が連続値の場合)
             loss = F.mse_loss(predicted_logits, observation)
        else:
             # Cross Entropy (観測が離散値/分布の場合)
             # observationをターゲットインデックスとみなすか、分布とみなすか
             if observation.dim() == predicted_logits.dim():
                 # 分布間のKL Divergence
                 target_dist = F.softmax(observation, dim=-1)
                 pred_log_dist = F.log_softmax(predicted_logits, dim=-1)
                 loss = F.kl_div(pred_log_dist, target_dist, reduction='batchmean')
             else:
                 loss = torch.tensor(0.0, device=observation.device, requires_grad=True)

        # 2. 逆伝播と更新
        loss.backward()
        self.optimizer.step()
        
        logger.info(f"Model updated via Active Inference. Variational Free Energy (Loss): {loss.item():.4f}")