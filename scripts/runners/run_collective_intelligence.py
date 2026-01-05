from snn_research.collective.liquid_democracy import LiquidDemocracyProtocol, Proposal
import sys
import os
import random
import logging

# Ensure project root is in path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_simulation():
    logger.info(
        "=== Starting Collective Intelligence Simulation (Liquid Democracy) ===")

    ld = LiquidDemocracyProtocol(decay_factor=0.95)

    # Define agents with different skills per topic
    agents = {
        "Math_Expert": {"Math": 0.95, "Art": 0.50, "type": "Expert (Math)"},
        "Art_Expert": {"Math": 0.50, "Art": 0.95, "type": "Expert (Art)"},
        "Generalist_1": {"Math": 0.70, "Art": 0.70, "type": "Generalist"},
        "Newbie_1": {"Math": 0.50, "Art": 0.50, "type": "Newbie"},
    }

    # Register agents
    for aid in agents:
        ld.register_agent(aid, initial_reputation=1.0, topics=["Math", "Art"])

    logger.info(f"Initialized {len(agents)} agents.")

    rounds = 10
    score_history = []

    # Alternate topics
    topics = ["Math", "Art"]

    for r in range(rounds):
        topic = topics[r % 2]
        logger.info(f"\n--- Round {r+1} (Topic: {topic}) ---")

        # Create proposals (A vs B)
        # Prop A is always correct in this sim
        prop_a = Proposal(
            id=f"prop_{r}_A", description=f"Option A (Correct {topic})", content="A", correct_choice="A", topic=topic)
        prop_b = Proposal(id=f"prop_{r}_B",
                          description="Option B", content="B", topic=topic)

        proposals = [prop_a, prop_b]
        round_votes = []

        for aid, stats in agents.items():
            skill = stats.get(topic, 0.5)

            # Logic: Determine choice
            is_correct = random.random() < skill
            choice_proposal = prop_a if is_correct else prop_b

            # Confidence based on skill
            confidence = skill * (0.8 + 0.4 * random.random())
            confidence = min(confidence, 1.0)

            delegate_threshold = 0.6

            # Identify best agent for THIS topic
            sorted_reps = ld.get_leaderboard(topic=topic)
            if not sorted_reps:
                best_agent = None
            else:
                best_agent = sorted_reps[0][0]
                # If self is best, pick second best? Or just vote.
                if best_agent == aid and len(sorted_reps) > 1:
                    best_agent = sorted_reps[1][0]
                elif best_agent == aid:
                    best_agent = None  # No one better to delegate to

            if confidence < delegate_threshold and best_agent:
                # Delegate
                try:
                    # ld.delegate(aid, best_agent, topic=topic)
                    # Use cast_vote for delegation recording
                    v = ld.cast_vote(voter_id=aid, proposal_id="DELEGATION",
                                     approve=False, confidence=0.0, delegate_to=best_agent, topic=topic)
                    round_votes.append(v)
                    logger.info(
                        f"  {aid} delegated to -> {best_agent} (Topic: {topic})")
                except ValueError:
                    pass
            else:
                # Vote
                v = ld.cast_vote(
                    voter_id=aid, proposal_id=choice_proposal.id, approve=True, confidence=confidence, topic=topic)
                round_votes.append(v)

        # Tally
        results = ld.tally_votes(proposals, round_votes)
        logger.info(f"  Results: {results}")

        if not results:
            winner_id = "None"
        else:
            winner_id = max(results, key=results.get)

        logger.info(f"  Consensus ID: {winner_id}")

        if winner_id == prop_a.id:
            logger.info("  >> SUCCESS")
            score_history.append(1)
            # Update reputation for this topic
            ld.update_reputation(winner_id, 1.0, topic=topic)
        else:
            logger.info("  >> FAILURE")
            score_history.append(0)

        # Show Top Reputations for this topic
        top_3 = ld.get_leaderboard(topic=topic)[:3]
        logger.info(f"  Top Reputations ({topic}): {top_3}")

    accuracy = sum(score_history) / len(score_history)
    logger.info("\n=== Simulation Complete ===")
    logger.info(f"Final Collective Accuracy: {accuracy * 100:.1f}%")


if __name__ == "__main__":
    run_simulation()
