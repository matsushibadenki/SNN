# ファイルパス: snn_research/collective/liquid_democracy.py
# Title: Liquid Democracy Protocol for SNN Agents
# Description:
#   ROADMAP Phase 4 Step 1: 集合知プロトコルの実装。
#   - Proposal: エージェントが提案する行動や判断。
#   - Vote: 賛否、信頼度重み付き投票。
#   - Delegation: 自信がない場合にエキスパートへ投票権を委ねる仕組み。

import uuid
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

@dataclass
class Proposal:
    """エージェントによる提案（行動計画や推論結果）"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    proposer_id: str = ""
    content: Any = None
    description: str = ""
    timestamp: float = 0.0

@dataclass
class Vote:
    """投票オブジェクト"""
    voter_id: str
    proposal_id: str
    approve: bool
    confidence: float # 0.0 - 1.0 (ニューロンの発火率やエントロピーから算出)
    delegated_to: Optional[str] = None # 委任先のAgent ID

class LiquidDemocracyProtocol:
    """
    流動的民主主義に基づく合意形成エンジン。
    """
    def __init__(self):
        # エージェントの評判スコア (Reputation Score)
        # 成功体験に基づいて変動する。初期値 1.0
        self.reputations: Dict[str, float] = {}
        self.vote_history: List[Vote] = []

    def register_agent(self, agent_id: str):
        if agent_id not in self.reputations:
            self.reputations[agent_id] = 1.0
            logger.info(f"🗳️ Agent registered: {agent_id}")

    def cast_vote(self, voter_id: str, proposal_id: str, approve: bool, confidence: float, delegate_to: Optional[str] = None) -> Vote:
        """投票を行う（または委任する）"""
        vote = Vote(
            voter_id=voter_id,
            proposal_id=proposal_id,
            approve=approve,
            confidence=confidence,
            delegated_to=delegate_to
        )
        self.vote_history.append(vote)
        
        type_str = f"Delegated -> {delegate_to}" if delegate_to else f"{'Approve' if approve else 'Reject'} (Conf: {confidence:.2f})"
        logger.debug(f"🗳️ Vote cast by {voter_id}: {type_str}")
        return vote

    def tally_votes(self, proposals: List[Proposal], votes: List[Vote]) -> Dict[str, float]:
        """
        投票を集計し、各提案のスコアを計算する。
        Score = Σ (Reputation * Confidence * Direction)
        委任（Delegation）がある場合、投票権を委任先に譲渡する。
        """
        scores = {p.id: 0.0 for p in proposals}
        
        # 委任チェーンの解決 (簡易実装: 1段階のみ)
        # {delegator: delegatee}
        delegations = {v.voter_id: v.delegated_to for v in votes if v.delegated_to is not None}
        
        # 実投票（委任していない票）の処理
        direct_votes = [v for v in votes if v.delegated_to is None]
        
        for vote in direct_votes:
            if vote.proposal_id not in scores:
                continue
                
            # 基本投票力 = 評判スコア
            voting_power = self.reputations.get(vote.voter_id, 1.0)
            
            # 委任分の加算
            # この投票者に委任している他のエージェントを探す
            for delegator_id, delegatee_id in delegations.items():
                if delegatee_id == vote.voter_id:
                    # 委任者の評判スコアを加算 (信頼度は委任先のものを適用するか、委任者のものを使うか議論があるが、ここでは委任先の判断を信頼するためVoting Powerごと渡す)
                    delegated_power = self.reputations.get(delegator_id, 1.0)
                    voting_power += delegated_power
                    logger.debug(f"  ↳ Delegated power (+{delegated_power:.1f}) from {delegator_id} to {vote.voter_id}")

            # スコア計算: Power * Confidence * (+1/-1)
            direction = 1.0 if vote.approve else -1.0
            impact = voting_power * vote.confidence * direction
            
            scores[vote.proposal_id] += impact
            
        return scores

    def update_reputation(self, winning_proposal_id: str, feedback_score: float):
        """
        結果に基づいて評判を更新する（強化学習的側面）。
        良い提案に賛成したエージェント -> 評判アップ
        悪い提案に賛成したエージェント -> 評判ダウン
        """
        # 今回の提案に関する投票を抽出
        relevant_votes = [v for v in self.vote_history if v.proposal_id == winning_proposal_id]
        
        for vote in relevant_votes:
            # 委任している場合は委任先の判断責任になるため、ここでは直接投票者のみ更新（簡易化）
            if vote.delegated_to:
                continue
                
            if vote.approve:
                # 賛成していた場合、フィードバック(正/負)をそのまま反映
                delta = feedback_score * 0.1
            else:
                # 反対していた場合、フィードバックの逆を反映 (悪い提案を却下したら評価アップ)
                delta = -feedback_score * 0.1
            
            old_rep = self.reputations.get(vote.voter_id, 1.0)
            new_rep = max(0.1, min(5.0, old_rep + delta)) # 0.1 ~ 5.0 の範囲
            self.reputations[vote.voter_id] = new_rep
            
        # 履歴クリア（デモ用）
        self.vote_history = []
        logger.info("📈 Reputation updated based on feedback.")

    def get_leaderboard(self):
        return sorted(self.reputations.items(), key=lambda x: x[1], reverse=True)