# ファイルパス: snn_research/cognitive_architecture/theory_of_mind.py
# 日本語タイトル: 心の理論 (Theory of Mind) モジュール
# 目的・内容:
#   ROADMAP Phase 3.1 対応。
#   他者の信念、意図、欲求を推論・シミュレーションするモジュール。
#   再帰的推論 (I believe that you believe...) をサポート。

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, field

from .global_workspace import GlobalWorkspace
from .rag_snn import RAGSystem

logger = logging.getLogger(__name__)


@dataclass
class AgentModel:
    """他者エージェントのメンタルモデル"""
    agent_id: str
    beliefs: Dict[str, float] = field(
        default_factory=dict)  # Fact -> Confidence
    goals: List[str] = field(default_factory=list)
    last_action: Optional[str] = None
    trust_level: float = 0.5


class TheoryOfMind:
    """
    心の理論 (ToM) エンジン。
    他者の視点をシミュレートし、行動を予測する。
    """

    def __init__(
        self,
        workspace: GlobalWorkspace,
        rag_system: RAGSystem,
        simulation_depth: int = 2
    ):
        self.workspace = workspace
        self.rag_system = rag_system
        self.simulation_depth = simulation_depth

        # エージェントモデルのデータベース
        self.agent_models: Dict[str, AgentModel] = {}

        # ワークスペースの購読
        self.workspace.subscribe(self.handle_broadcast)

        logger.info(
            f"🧠 Theory of Mind module initialized (Depth: {simulation_depth})")

    def get_or_create_agent(self, agent_id: str) -> AgentModel:
        if agent_id not in self.agent_models:
            self.agent_models[agent_id] = AgentModel(agent_id=agent_id)
            logger.info(f"👤 New agent model created: {agent_id}")
        return self.agent_models[agent_id]

    def update_agent_belief(self, agent_id: str, fact: str, confidence: float):
        """他者の信念を更新する"""
        agent = self.get_or_create_agent(agent_id)
        agent.beliefs[fact] = confidence
        logger.debug(
            f"🧠 Updated belief for {agent_id}: {fact} (conf={confidence:.2f})")

    def infer_intent(self, agent_id: str, action: str, context: str) -> str:
        """
        他者の行動から意図を推論する。
        簡易的な実装: アクションとコンテキストからゴールを推測。
        """
        agent = self.get_or_create_agent(agent_id)
        agent.last_action = action

        # 本来はLLMや学習済みモデルで推論する箇所
        # ここではルールベースの簡易推論
        intent = "unknown"
        if "asking" in action or "question" in action:
            intent = "information_seeking"
        elif "attacking" in action or "threat" in action:
            intent = "hostile"
        elif "helping" in action or "sharing" in action:
            intent = "cooperative"

        logger.info(
            f"🤔 Inferred intent of {agent_id} for action '{action}': {intent}")

        # 意図をワークスペースに投稿
        self.workspace.upload_to_workspace(
            source="theory_of_mind",
            data={
                "type": "intent_inference",
                "target_agent": agent_id,
                "action": action,
                "inferred_intent": intent
            },
            salience=0.7
        )
        return intent

    def simulate_other(self, agent_id: str, context: str) -> str:
        """
        [Simulation] 他者の視点に立って、その反応を予測する。
        """
        agent = self.get_or_create_agent(agent_id)

        # 他者の信念に基づいたコンテキストを作成
        simulated_context = f"Context: {context}\n"
        simulated_context += f"Agent {agent_id} believes:\n"
        for fact, conf in agent.beliefs.items():
            if conf > 0.5:
                simulated_context += f"- {fact}\n"

        # 本来はここでLLM等を呼び出し、simulated_contextに対する反応を生成
        # 簡易シミュレーション
        predicted_reaction = f"Agent {agent_id} acts based on beliefs: {list(agent.beliefs.keys())}"

        logger.info(
            f"🎭 Simulated perspective of {agent_id}: {predicted_reaction}")
        return predicted_reaction

    def handle_broadcast(self, source: str, data: Any):
        """
        意識に上った情報から、社会的シグナルを検出する。
        """
        if not isinstance(data, dict):
            return

        # 会話やインタラクションの検出
        if data.get("type") == "interaction" or source == "auditory_cortex":
            agent_id = data.get("agent_id", "unknown_user")
            content = data.get("content", "")

            # 信念の更新 (発話内容は相手がそう信じていると仮定)
            if content:
                self.update_agent_belief(agent_id, f"said: {content}", 0.8)
                self.infer_intent(agent_id, content, "conversation")
