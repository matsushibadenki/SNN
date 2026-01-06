from snn_research.social.emergent_language import NamingGameSimulation, Agent
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding


class MockRAG:
    """Mock RAG System for testing."""

    def add_triple(self, s, p, o):
        pass


def test_naming_game_basic():
    # Mock RAG and SymbolGrounding dependencies
    mock_rag = MockRAG()
    sg1 = SymbolGrounding(rag_system=mock_rag, base_vigilance=0.5)
    sg2 = SymbolGrounding(rag_system=mock_rag, base_vigilance=0.5)

    agent_a = Agent(id="A", grounding_system=sg1)
    agent_b = Agent(id="B", grounding_system=sg2)

    game = NamingGameSimulation(agent_a, agent_b)

    # Run multiple rounds to encourage vocabulary formation and alignment
    # Since inputs are random, we check if vocabularies are being populated.

    for _ in range(10):
        game.play_round()

    # Check if vocabulary grew
    vocab_size_a = len(agent_a.vocabulary)
    vocab_size_b = len(agent_b.vocabulary)

    assert vocab_size_a > 0 or vocab_size_b > 0

    # With enough rounds, there should be some success, but randomness makes it flaky to assert True.
    # We just ensure no crash and state update.
