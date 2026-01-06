import logging
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.social.emergent_language import Agent, NamingGameSimulation

logging.basicConfig(level=logging.INFO)


def run_simulation():
    print("🚀 Starting Naming Game Simulation...")

    # Setup mocks
    rag_a = RAGSystem()
    rag_b = RAGSystem()
    grounding_a = SymbolGrounding(rag_a)
    grounding_b = SymbolGrounding(rag_b)

    agent_a = Agent("Alice", grounding_a)
    agent_b = Agent("Bob", grounding_b)

    game = NamingGameSimulation(agent_a, agent_b)

    # Run loop
    num_rounds = 50
    for i in range(num_rounds):
        game.play_round()

    print(
        f"🏁 Simulation Finished. Success Rate: {game.success_count}/{game.total_rounds} ({game.success_count/game.total_rounds*100:.1f}%)")

    # Verify vocab size
    print(f"Alice Vocab Size: {len(agent_a.vocabulary)}")
    print(f"Bob Vocab Size: {len(agent_b.vocabulary)}")

    with open("naming_game_result.txt", "w") as f:
        f.write(
            f"Success Rate: {game.success_count/game.total_rounds*100:.1f}%\n")
        f.write(f"Alice Vocab: {len(agent_a.vocabulary)}\n")


if __name__ == "__main__":
    run_simulation()
