# ファイルパス: snn_research/collective/liquid_democracy.py
# 日本語タイトル: Liquid Democracy Protocol (Type Safe)
# 修正内容: Tensorとfloatの型不一致エラーを修正 (.item()の追加)。

import torch
import logging
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass

from snn_research.agent.synesthetic_agent import SynestheticAgent
from snn_research.social.theory_of_mind import TheoryOfMindModule

logger = logging.getLogger(__name__)


@dataclass
class Vote:
    agent_id: str
    decision: int  # 0 or 1
    weight: float = 1.0


@dataclass
class Proposal:
    """
    投票対象となる提案や課題。
    """
    id: str
    content: Any  # torch.Tensor or Text
    description: str = ""


class LiquidDemocracyProtocol:
    """
    流動的民主主義プロトコル。
    ToMを用いた委任(Delegation)と、加重投票(Weighted Voting)を管理する。
    """

    def __init__(self, agents: Dict[str, SynestheticAgent], toms: Dict[str, TheoryOfMindModule]):
        self.agents = agents
        self.toms = toms  # AgentID -> TheoryOfMindModule
        if agents:
            self.device = next(iter(agents.values())).device
        else:
            self.device = 'cpu'

    def conduct_vote(self, task_input: torch.Tensor, ground_truth: Optional[int] = None) -> Dict[str, Any]:
        """
        投票サイクルを実行する。
        """
        # 1. 各エージェントの初期判断と自信度
        initial_decisions: Dict[str, Tuple[int, float]] = {}

        for agent_id, agent in self.agents.items():
            # 入力形式の調整
            obs = {'vision': task_input.unsqueeze(
                0) if task_input.dim() == 1 else task_input}

            with torch.no_grad():
                # Agentのstep戻り値は {"action_pred": Tensor, ...} などを想定
                # 簡易的に action_pred が Tensor(1, dim) で返ると仮定
                result = agent.step(obs)

                # 結果の取り出し（辞書またはTensorに対応）
                if isinstance(result, dict):
                    action = result.get("action_pred", torch.tensor([[0.0]]))
                else:
                    action = result

                if isinstance(action, torch.Tensor):
                    val = action.mean().item()  # 平均値で簡易判定
                else:
                    val = 0.0

            decision = 1 if val > 0 else 0
            confidence = abs(val)
            initial_decisions[agent_id] = (decision, confidence)

        # 2. 委任フェーズ (Delegation)
        vote_powers: Dict[str, float] = {
            aid: 1.0 for aid in self.agents.keys()}
        delegation_map: Dict[str, str] = {}  # from -> to

        for agent_id, (my_dec, my_conf) in initial_decisions.items():
            # 自信が低い場合は委任を検討
            if my_conf < 0.3:
                best_target = None
                max_trust = -1.0

                tom = self.toms.get(agent_id)
                if tom is None:
                    continue

                for other_id in self.agents.keys():
                    if other_id == agent_id:
                        continue

                    # ToMによる信頼度予測 (Tensor -> float)
                    trust_tensor = tom.predict_action(other_id)
                    trust_val = trust_tensor.item()

                    if trust_val > max_trust:
                        max_trust = trust_val
                        best_target = other_id

                # 信頼できる相手がいれば委任
                if best_target and max_trust > 0.6:
                    delegation_map[agent_id] = best_target
                    logger.debug(
                        f"🔄 {agent_id} delegates to {best_target} (Trust: {max_trust:.2f})")

        # 票の移動 (Single Hop)
        final_voters = []
        for agent_id in self.agents.keys():
            if agent_id in delegation_map:
                target = delegation_map[agent_id]
                # 委任先の票を増やす
                if target in vote_powers:
                    vote_powers[target] += vote_powers[agent_id]
                vote_powers[agent_id] = 0.0
            else:
                final_voters.append(agent_id)

        # 3. 集計 (Aggregation)
        weighted_sum = 0.0
        total_power = 0.0

        for voter_id in final_voters:
            decision, _ = initial_decisions[voter_id]
            power = vote_powers[voter_id]

            weighted_sum += decision * power
            total_power += power

        final_score = weighted_sum / total_power if total_power > 0 else 0.0
        final_decision = 1 if final_score >= 0.5 else 0

        # 4. 社会的学習 (Social Learning)
        is_correct = None
        if ground_truth is not None:
            is_correct = (final_decision == ground_truth)

            for agent_id in self.agents.keys():
                target_id = delegation_map.get(agent_id, agent_id)

                # 委任先の判断が正しかったか評価
                target_dec, _ = initial_decisions[target_id]
                target_correct = (target_dec == ground_truth)

                outcome_val = 1.0 if target_correct else 0.0

                # ToMモデルの更新
                if agent_id in self.toms:
                    self.toms[agent_id].update_model(target_id, outcome_val)

        return {
            'consensus_decision': final_decision,
            'vote_ratio': final_score,
            'delegation_count': len(delegation_map),
            'correct': is_correct
        }
