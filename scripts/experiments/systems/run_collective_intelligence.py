# ファイルパス: scripts/runners/run_collective_intelligence.py
# Title: Phase 4 Demo - Collective Intelligence Swarm (Debug Version)
# Description:
#   複数のBrain Agentが協力して画像認識タスクの意思決定を行う。
#   Liquid Democracyを用いて、自信のないエージェントがエキスパートに投票を委任する様子を再現。
#   修正: 出力が表示されない問題に対処するため、PrintデバッグとStreamHandlerを追加。

import sys
import os
import time
import logging
import random
import traceback
from typing import List, Dict

# --- Immediate Debug Print ---
print(f"[DEBUG] Script process started. PID: {os.getpid()}")
sys.stdout.flush()

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

# ロギング設定 (標準出力へ強制的に流す)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CollectiveSwarm")

try:
    print("[DEBUG] Importing modules...")
    # Collective Intelligence Components
    from snn_research.collective.liquid_democracy import LiquidDemocracyProtocol, Proposal, Vote
    print("[DEBUG] Imports successful.")
except ImportError as e:
    print(f"[ERROR] Module import failed. Check file paths. Error: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Unexpected error during import: {e}")
    traceback.print_exc()
    sys.exit(1)

class SwarmAgent:
    """
    Brain v21を搭載した個別のエージェント。
    """
    def __init__(self, agent_id: str, role: str, expertise_level: float):
        self.id = agent_id
        self.role = role
        self.expertise = expertise_level # 0.0 - 1.0 (このタスクへの適性)
        self.internal_confidence = 0.0

    def perceive(self, task_difficulty: float) -> float:
        """
        タスクを観察し、自身の信頼度（Confidence）を計算する。
        """
        # ノイズや難易度、自身の専門性に基づく信頼度計算
        base_confidence = self.expertise - (task_difficulty * 0.5) + (random.random() * 0.2)
        self.internal_confidence = max(0.0, min(1.0, base_confidence))
        return self.internal_confidence

    def decide_vote(self, proposals: List[Proposal], protocol: LiquidDemocracyProtocol) -> Vote:
        """
        提案に対して投票するか、誰かに委任するかを決定する。
        """
        DELEGATION_THRESHOLD = 0.4
        
        # 自信が閾値以下の場合は、自分より評判の高いエキスパートに委任
        if self.internal_confidence < DELEGATION_THRESHOLD:
            leaderboard = protocol.get_leaderboard()
            my_rep = protocol.reputations.get(self.id, 0)
            
            for expert_id, rep in leaderboard:
                if expert_id != self.id and rep > my_rep:
                    return protocol.cast_vote(
                        voter_id=self.id,
                        proposal_id="DELEGATION", 
                        approve=False,
                        confidence=0.0,
                        delegate_to=expert_id
                    )
        
        # 自信がある場合は投票
        target_proposal = proposals[0] # デフォルト
        # シミュレーション: 専門性が高いほど正解(0番目)を選びやすい
        if random.random() < self.expertise:
            target_proposal = proposals[0] 
        else:
            if len(proposals) > 1:
                target_proposal = proposals[1]

        return protocol.cast_vote(
            voter_id=self.id,
            proposal_id=target_proposal.id,
            approve=True,
            confidence=self.internal_confidence
        )

def run_collective_demo():
    logger.info(">>> Starting Phase 4 Collective Intelligence Demo...")
    sys.stdout.flush()
    
    try:
        protocol = LiquidDemocracyProtocol()
        
        # 1. スワームの生成
        logger.info("Creating swarm agents...")
        agents = [
            SwarmAgent("Agent_Alpha", "Generalist", 0.6),
            SwarmAgent("Agent_Beta",  "Visual_Expert", 0.95), # エキスパート
            SwarmAgent("Agent_Gamma", "Generalist", 0.5),
            SwarmAgent("Agent_Delta", "Newbie", 0.3),
            SwarmAgent("Agent_Epsilon","Newbie", 0.35)
        ]
        
        for ag in agents:
            protocol.register_agent(ag.id)
            
        # シナリオ: 難易度の高いタスク
        task_difficulty = 0.7 
        logger.info(f"🧩 New Task Incoming (Difficulty: {task_difficulty:.2f})")
        
        # 2. 提案の生成
        proposals = [
            Proposal(proposer_id="System", content="Action_A (Correct)", description="Avoid Obstacle"),
            Proposal(proposer_id="System", content="Action_B (Risky)",   description="Go Straight")
        ]
        logger.info("📋 Proposals on table:")
        for p in proposals:
            logger.info(f"  - [{p.id}] {p.description}")

        # 3. 各エージェントの思考と投票
        logger.info("\n--- Voting Phase ---")
        votes = []
        
        # 感知 & 投票
        for ag in agents:
            ag.perceive(task_difficulty)
            vote = ag.decide_vote(proposals, protocol)
            votes.append(vote)
            
            # 詳細ログ
            action = f"Delegated to {vote.delegated_to}" if vote.delegated_to else f"Voted (Conf: {vote.confidence:.2f})"
            logger.info(f"  🤖 {ag.id} [{ag.role}]: {action}")

        # 4. 集計 (Liquid Democracy)
        logger.info("\n--- Tallying Votes (Liquid Democracy) ---")
        scores = protocol.tally_votes(proposals, votes)
        
        for pid, score in scores.items():
            # Proposal IDから説明を検索
            try:
                p_desc = next(p.description for p in proposals if p.id == pid)
                logger.info(f"  Proposal '{p_desc}': Score = {score:.2f}")
            except StopIteration:
                pass # 委任票などはスキップ

        # 決定
        if not scores:
            logger.error("No valid votes found.")
            return

        winner_id = max(scores, key=scores.get)
        winner = next(p for p in proposals if p.id == winner_id)
        logger.info(f"🏆 Winning Proposal: {winner.description}")

        # 5. フィードバックと学習
        # 正解は Action_A (Avoid Obstacle)
        if winner.content == "Action_A (Correct)":
            feedback = 1.0
            logger.info("✅ Outcome: SUCCESS. The swarm made the right choice.")
        else:
            feedback = -1.0
            logger.info("❌ Outcome: FAILURE. The swarm crashed.")
            
        protocol.update_reputation(winner_id, feedback)
        
        # 最終的な評判スコア
        logger.info("\n--- Updated Reputation Leaderboard ---")
        for agent_id, rep in protocol.get_leaderboard():
            logger.info(f"  {agent_id}: {rep:.2f}")
            
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_collective_demo()