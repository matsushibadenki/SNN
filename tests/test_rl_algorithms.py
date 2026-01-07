import torch
import numpy as np
from snn_research.training.rl.spike_ppo import SpikePPO
from snn_research.training.rl.spike_sac import SpikeSAC, ReplayBuffer


class DummyBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []


def test_ppo_step():
    state_dim = 4
    action_dim = 2
    agent = SpikePPO(state_dim, action_dim, is_continuous=True)

    # Select action
    state = torch.randn(1, state_dim)
    action, log_prob = agent.select_action(state)
    assert action.shape == (1, action_dim)

    # Update
    buffer = DummyBuffer()
    buffer.states = [state]
    buffer.actions = [action]
    buffer.log_probs = [log_prob]
    buffer.rewards = [1.0]
    buffer.is_terminals = [False]

    # Run update (should not crash)
    agent.update(buffer)


def test_sac_step():
    state_dim = 4
    action_dim = 2
    agent = SpikeSAC(state_dim, action_dim)

    # Select action
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    assert action.shape == (action_dim,)

    # Update
    buffer = ReplayBuffer(100)
    buffer.push(state, action, 1.0, state, False)

    # Run update (needs batch size)
    agent.update(buffer, batch_size=1)
