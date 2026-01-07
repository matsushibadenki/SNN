import torch
from snn_research.adaptive.active_inference_agent import ActiveInferenceAgent


def test_active_inference_forward():
    state_dim = 4
    obs_dim = 2
    action_dim = 2
    agent = ActiveInferenceAgent(state_dim, obs_dim, action_dim)

    obs = torch.randn(1, obs_dim)
    prev_action = torch.zeros(1, action_dim)

    action, free_energy = agent(obs, prev_action)

    assert action.shape == (1, action_dim)
    assert isinstance(free_energy, float)


def test_active_inference_update():
    state_dim = 4
    obs_dim = 2
    action_dim = 2
    agent = ActiveInferenceAgent(state_dim, obs_dim, action_dim)

    obs = torch.randn(1, obs_dim)
    action = torch.randn(1, action_dim)
    next_obs = torch.randn(1, obs_dim)

    # Check if parameters change/update runs without error
    # We can't easily check param change without cloning, but ensuring no error is a good start
    agent.update_model(obs, action, next_obs)
