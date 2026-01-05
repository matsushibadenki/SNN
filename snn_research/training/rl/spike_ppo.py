# ファイルパス: snn_research/training/rl/spike_ppo.py
# Title: Spike PPO (Proximal Policy Optimization)
# Description:
#   スパイキングニューラルネットワーク向けのPPO実装。
#   連続値アクションと離散アクションの両方をサポートする基盤。

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Any


class ActorCriticSNN(nn.Module):
    """
    SNNベースのActor-Criticネットワーク (簡易実装)
    実際にはLIFモデル等を使うが、ここではインターフェース定義としてMLPで代用し、
    将来的にSNNモジュールに差し替え可能な構造とする。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, is_continuous: bool = True):
        super().__init__()
        self.is_continuous = is_continuous

        # 共通層 (Feature Extractor)
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor: 行動決定
        if is_continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(
                torch.zeros(1, action_dim))  # 分散は学習パラメータ
        else:
            self.actor_logits = nn.Linear(hidden_dim, action_dim)

        # Critic: 価値推定
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        """
        Returns:
            action_dist: 行動分布 (Normal or Categorical)
            state_value: 状態価値 V(s)
        """
        features = self.shared_layer(state)
        value = self.critic(features)

        dist: Any
        if self.is_continuous:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_log_std).expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
        else:
            logits = self.actor_logits(features)
            dist = torch.distributions.Categorical(logits=logits)

        return dist, value


class SpikePPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        is_continuous: bool = True
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCriticSNN(
            state_dim, action_dim, is_continuous=is_continuous)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCriticSNN(
            state_dim, action_dim, is_continuous=is_continuous)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.MSELoss()

    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        状態から行動を選択し、log_probと共に返す (推論用)
        """
        with torch.no_grad():
            dist, _ = self.policy_old(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            if self.policy.is_continuous:
                return action, log_prob.sum(dim=-1)  # Continuousの場合sumをとる
            else:
                return action, log_prob

    def update(self, buffer):
        """
        バッファに溜まった経験を用いて学習を行う
        """
        # バッファから全データを展開 (PPOはOn-Policyなので溜めた分を全部使う)
        # bufferの実装依存だが、ここではリストで渡されると仮定
        states = torch.stack(buffer.states).detach()
        actions = torch.stack(buffer.actions).detach()
        log_probs = torch.stack(buffer.log_probs).detach()
        rewards = buffer.rewards
        is_terminals = buffer.is_terminals

        # 割引報酬和 (Monte Carlo Estimate)
        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)

        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32)
        if torch.cuda.is_available():  # Device対応 (簡易)
            rewards_to_go = rewards_to_go.to(states.device)

        # Normalize rewards
        if len(rewards_to_go) > 1:
            rewards_to_go = (rewards_to_go - rewards_to_go.mean()
                             ) / (rewards_to_go.std() + 1e-7)
        else:
            rewards_to_go = rewards_to_go - rewards_to_go.mean()  # Just center it

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values
            dist, state_values = self.policy(states)

            if self.policy.is_continuous:
                action_logprobs = dist.log_prob(actions).sum(dim=-1)
                dist_entropy = dist.entropy().sum(dim=-1)
            else:
                action_logprobs = dist.log_prob(actions)
                dist_entropy = dist.entropy()

            state_values = torch.squeeze(state_values)

            # Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(action_logprobs - log_probs.detach())

            # Surrogate Loss
            if state_values.dim() == 0:  # Handle scalar case
                state_values = state_values.unsqueeze(0)

            advantages = rewards_to_go - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages

            # Total Loss = -ActorLoss + 0.5*CriticLoss - 0.01*Entropy
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.loss_fn(state_values, rewards_to_go) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()

            # Gradient Clipping
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

            self.optimizer.step()

        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
