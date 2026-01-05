from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class Proposal:
    """Represents a decision or task to be voted on."""
    id: str
    description: str
    content: Optional[str] = None
    topic: str = "general"  # Topic for expertise routing
    proposer_id: Optional[str] = None
    choices: List[str] = field(default_factory=list)
    correct_choice: Optional[str] = None


@dataclass
class Vote:
    """Represents a vote cast by an agent."""
    agent_id: str
    proposal_id: str
    approve: bool
    confidence: float
    delegated_to: Optional[str] = None
    choice: Optional[str] = None  # To support multi-choice if needed


class LiquidDemocracyProtocol:
    """
    Manages the Liquid Democracy protocol, including voting, delegation,
    and reputation tracking.
    Enhanced to support Topic-based Reputation and Decay.
    """

    def __init__(self, decay_factor: float = 0.95) -> None:
        # Reputation is now Dict[topic, Dict[agent_id, score]]
        self.reputations: Dict[str, Dict[str, float]] = {}
        self.vote_history: List[Vote] = []
        # Map delegator -> topic -> delegatee (persistent)
        self.delegations: Dict[str, Dict[str, str]] = {}
        self.decay_factor = decay_factor

    def register_agent(self, agent_id: str, initial_reputation: float = 1.0, topics: Optional[List[str]] = None) -> None:
        if topics is None:
            topics = ["general"]

        for topic in topics:
            if topic not in self.reputations:
                self.reputations[topic] = {}
            self.reputations[topic][agent_id] = initial_reputation

    def get_reputation(self, agent_id: str, topic: str) -> float:
        # Default to 1.0 if unknown
        return self.reputations.get(topic, {}).get(agent_id, 1.0)

    def get_leaderboard(self, topic: str = "general") -> List[Tuple[str, float]]:
        if topic not in self.reputations:
            return []
        return sorted(self.reputations[topic].items(), key=lambda x: x[1], reverse=True)

    def cast_vote(self, voter_id: str, proposal_id: str, approve: bool, confidence: float, delegate_to: Optional[str] = None, topic: str = "general") -> Vote:
        """
        Records a vote or delegation.
        """
        v = Vote(
            agent_id=voter_id,
            proposal_id=proposal_id,
            approve=approve,
            confidence=confidence,
            delegated_to=delegate_to
        )
        self.vote_history.append(v)

        # Also update persistent delegations map if this vote is a delegation
        if delegate_to:
            if voter_id not in self.delegations:
                self.delegations[voter_id] = {}
            self.delegations[voter_id][topic] = delegate_to

        return v

    def delegate(self, delegator_id: str, delegatee_id: str, topic: str = "general") -> None:
        if delegator_id == delegatee_id:
            raise ValueError("Cannot delegate to self.")
        if delegator_id not in self.delegations:
            self.delegations[delegator_id] = {}
        self.delegations[delegator_id][topic] = delegatee_id

    def tally_votes(self, proposals: List[Proposal], votes: List[Vote]) -> Dict[str, float]:
        """
        Tally votes considering delegations and topics.
        """
        scores: Dict[str, float] = {p.id: 0.0 for p in proposals}

        # Map proposal IDs to their topics for easy lookup
        proposal_topics = {p.id: p.topic for p in proposals}

        relevant_proposal_ids = {p.id for p in proposals}
        active_voters: Set[str] = set()

        # Map: agent_id -> (proposal_id, confidence)
        direct_votes: Dict[str, Tuple[str, float]] = {}

        # Handle Delegation Votes (Transient for this tally, but backed by persistent)
        # We need a local delegation map that respects topics.
        # Structure: delegator -> topic -> delegatee
        local_delegations = {k: v.copy() for k, v in self.delegations.items()}

        for v in votes:
            # Determining topic for delegation vote is tricky if proposal_id is "DELEGATION"
            # In our simulation, we treat "DELEGATION" votes as updates to the persistent map mostly.
            # But here we need to know the TOPIC of the delegation if it's dynamic.
            # For simplicity, if v.delegated_to is set, we assume it applies to the topic of the current context
            # BUT wait, cast_vote doesn't store topic in Vote.
            # We should probably infer it from context or add topic to Vote?
            # Ideally, a Vote for a specific Proposal knows its topic.
            # A pure Delegation vote might need a topic field or we assume it matches the current round's topic.

            # Let's check if it's a direct vote
            if v.proposal_id in relevant_proposal_ids and v.approve:
                direct_votes[v.agent_id] = (v.proposal_id, v.confidence)
                active_voters.add(v.agent_id)

        # 2. Calculate Effective Weight for each Direct Voter

        def get_weight(agent_id: str, topic: str, visited: Set[str]) -> float:
            if agent_id in visited:
                return 0.0
            visited.add(agent_id)

            # Base reputation for this topic
            w = self.get_reputation(agent_id, topic)

            # Plus delegations from others on THIS topic
            # Find all X where local_delegations[X][topic] == agent_id
            # AND X did NOT vote directly
            for delegator, topic_map in local_delegations.items():
                if topic_map.get(topic) == agent_id:
                    if delegator not in active_voters:
                        w += get_weight(delegator, topic, visited.copy())
            return w

        # 3. Sum up scores
        for voter, (pid, conf) in direct_votes.items():
            topic = proposal_topics.get(pid, "general")
            weight = get_weight(voter, topic, set())
            scores[pid] += weight * conf

        return scores

    def update_reputation(self, winner_proposal_id: str, feedback_score: float, topic: str = "general") -> None:
        """
        Update reputation of agents who supported the winning proposal.
        Applies decay first.
        """
        # 1. Decay all reputations in this topic
        if topic in self.reputations:
            for agent in self.reputations[topic]:
                self.reputations[topic][agent] *= self.decay_factor

        # 2. Reward winners
        # Find who voted for this proposal
        supporters = []
        for v in self.vote_history:
            if v.proposal_id == winner_proposal_id and v.approve:
                supporters.append(v.agent_id)

        # Update
        for agent_id in supporters:
            if topic not in self.reputations:
                self.reputations[topic] = {}
            current_rep = self.reputations[topic].get(agent_id, 1.0)
            self.reputations[topic][agent_id] = current_rep + \
                (0.1 * feedback_score)
