import torch
from snn_research.social.theory_of_mind import TheoryOfMindModule


def test_tom_prediction():
    tom = TheoryOfMindModule(observation_dim=2, history_len=3)
    agent_id = "agent_bob"

    # 1. Observe some consistent actions (always [1.0, 1.0])
    for _ in range(5):
        action = torch.tensor([1.0, 1.0])
        tom.observe_agent(agent_id, action)

    # 2. Predict
    pred = tom.predict_action(agent_id)
    # Initial random weights, but it should return a float probability
    assert 0.0 <= pred <= 1.0

    # Check history structure
    assert len(tom.interaction_history[agent_id]) == 3  # maxlen=3
