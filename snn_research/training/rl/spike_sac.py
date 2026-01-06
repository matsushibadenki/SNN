# ファイルパス: snn_research/training/rl/spike_sac.py
# Title: Spike SAC (Soft Actor-Critic) with LIF Neurons
# Description:
#   連続値制御に適したSoft Actor-Criticアルゴリズムの実装。
#   LIFニューロンを用いたSNNベースのポリシーおよびQ関数ネットワークを採用。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List
from snn_research.core.neurons.lif_neuron import LIFNeuron


class ReplayBuffer:
    """簡易リプレイバッファ"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        import random
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


class SpikingSoftQNetwork(nn.Module):
    """SNNベースのQ関数 (Critic) x 2"""

    def __init__(self, state_dim, action_dim, hidden_dim=256, time_steps=4):
        super().__init__()
        self.time_steps = time_steps

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.lif1 = LIFNeuron(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = LIFNeuron(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.lif3 = LIFNeuron(hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.lif4 = LIFNeuron(hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        self.lif1.reset()
        self.lif2.reset()
        self.lif3.reset()
        self.lif4.reset()

        q1_accum = 0.0
        q2_accum = 0.0

        # Rate coding simulation
        for _ in range(self.time_steps):
            # Q1 Path
            sp1, _ = self.lif1(self.l1(xu))
            sp2, _ = self.lif2(self.l2(sp1))  # Deep SNN
            q1_accum += self.l3(sp2)

            # Q2 Path
            sp3, _ = self.lif3(self.l4(xu))
            sp4, _ = self.lif4(self.l5(sp3))
            q2_accum += self.l6(sp4)

        return q1_accum / self.time_steps, q2_accum / self.time_steps


class SpikingPolicyNetwork(nn.Module):
    """SNNベースのActor (Gaussian Policy)"""

    def __init__(self, state_dim, action_dim, hidden_dim=256, action_space=None, time_steps=4):
        super().__init__()
        self.time_steps = time_steps

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.lif1 = LIFNeuron(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = LIFNeuron(hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        self.lif1.reset()
        self.lif2.reset()

        out_accum = 0.0

        for _ in range(self.time_steps):
            sp1, _ = self.lif1(self.l1(state))
            sp2, _ = self.lif2(self.l2(sp1))
            out_accum += sp2

        # Average rate
        features = out_accum / self.time_steps

        mean = self.mean_linear(features)
        log_std = self.log_std_linear(features)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # for reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean


class SpikeSAC:
    def __init__(self, state_dim, action_dim, action_space=None, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.critic = SpikingSoftQNetwork(state_dim, action_dim)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target = SpikingSoftQNetwork(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = SpikingPolicyNetwork(
            state_dim, action_dim, action_space=action_space)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, memory, batch_size):
        if len(memory) < batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size)

        state_batch = torch.stack(
            list(map(lambda x: torch.tensor(x).float(), state_batch)))
        next_state_batch = torch.stack(
            list(map(lambda x: torch.tensor(x).float(), next_state_batch)))
        action_batch = torch.stack(
            list(map(lambda x: torch.tensor(x).float(), action_batch)))
        reward_batch = torch.tensor(reward_batch).float().unsqueeze(1)
        mask_batch = torch.tensor(mask_batch).float().unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(
                next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            min_qf_next_target = torch.min(
                qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + \
                (1 - mask_batch) * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)
