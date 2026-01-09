# ファイルパス: snn_research/collective/liquid_democracy.py
# 日本語タイトル: Liquid Democracy Protocol (LDP) Engine - Type Fixed
# 修正内容: 戻り値の型ヒントをAnyに変更し、Proposalクラスを追加。

import torch
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

# 既存のモジュールをインポート
from snn_research.agent.synesthetic_agent import SynestheticAgent
from snn_research.social.theory_of_mind import TheoryOfMindModule

logger = logging.getLogger(__name__)


@dataclass
class Vote:
    agent_id: str
    decision: int  # 0 or 1 (Binary decision for simplicity)
    weight: float = 1.0


@dataclass
class Proposal:
    """
    投票対象となる提案や課題を格納するデータクラス。
    他のスクリプト(run_unified_mission.py等)からの参照用に追加。
    """
    id: str
    content: Any  # torch.Tensor or Text
    description: str = ""


class LiquidDemocracyProtocol:
    """
    流動的民主主義プロトコル。

    Process:
    1. Proposal: 課題（入力データ）が提示される。
    2. Evaluation: 各エージェントが自身の自信度(Confidence)を評価。
    3. Delegation: 自信がないエージェントは、ToMを用いて「自分より詳しそうなエージェント」に委任する。
    4. Voting: 委任された票(Power)を持ったエージェントが投票する。
    5. Consensus: 加重多数決で最終決定を行う。
    """

    def __init__(self, agents: Dict[str, SynestheticAgent], toms: Dict[str, TheoryOfMindModule]):
        self.agents = agents
        self.toms = toms  # AgentID -> Its ToM Module
        if agents:
            self.device = next(iter(agents.values())).device
        else:
            self.device = 'cpu'

    def conduct_vote(self, task_input: torch.Tensor, ground_truth: Optional[int] = None) -> Dict[str, Any]:
        """
        1回の投票サイクルを実行する。

        Args:
            task_input: 判定対象のデータ (例: 画像特徴量)
            ground_truth: 正解ラベル (学習用、推論時はNone)
        Returns:
            metrics: {'accuracy': float, 'delegation_rate': float, 'consensus': float, 'correct': bool/None}
        """
        # 1. 各エージェントの初期判断と自信度
        initial_decisions = {}  # id -> (decision, confidence)

        for agent_id, agent in self.agents.items():
            # エージェントに思考させる (Brain v4)
            # 入力形式を整える (Brain v4は辞書入力を期待)
            # task_inputの次元によって vision/audio 等に割り振るが、ここでは vision と仮定
            obs = {'vision': task_input.unsqueeze(
                0) if task_input.dim() == 1 else task_input}

            with torch.no_grad():
                action = agent.step(obs)  # (1, ActionDim)
                val = action[0, 0].item()  # 1次元目を決定値とする

            decision = 1 if val > 0 else 0
            confidence = abs(val)
            initial_decisions[agent_id] = (decision, confidence)

        # 2. 委任フェーズ (Delegation Logic)
        vote_powers = {aid: 1.0 for aid in self.agents.keys()}  # 初期の持ち票は1
        delegation_map = {}  # from -> to

        for agent_id, (my_dec, my_conf) in initial_decisions.items():
            # 自信が閾値以下なら委任を検討
            if my_conf < 0.3:
                best_target = None
                max_trust = -1.0

                # ToMを使って他のエージェントの信頼度を確認
                tom = self.toms[agent_id]
                for other_id in self.agents.keys():
                    if other_id == agent_id:
                        continue

                    # ToMの predict_action は "相手が協力してくれる確率(0~1)" を返す
                    # これを「信頼度」として代用
                    trust = tom.predict_action(other_id)

                    if trust > max_trust:
                        max_trust = trust
                        best_target = other_id

                # 信頼できる相手がいれば委任
                if best_target and max_trust > 0.6:
                    delegation_map[agent_id] = best_target
                    logger.debug(
                        f"🔄 {agent_id} delegates to {best_target} (Trust: {max_trust:.2f})")

        # 票の移動処理 (1ホップのみ実装)
        final_voters = []
        for agent_id in self.agents.keys():
            if agent_id in delegation_map:
                target = delegation_map[agent_id]
                vote_powers[target] += vote_powers[agent_id]
                vote_powers[agent_id] = 0  # 委任したので自分の行使権は消滅
            else:
                final_voters.append(agent_id)

        # 3. 集計 (Aggregation)
        weighted_sum = 0.0
        total_power = 0.0

        for voter_id in final_voters:
            decision, _ = initial_decisions[voter_id]
            power = vote_powers[voter_id]

            # 0 or 1
            weighted_sum += decision * power
            total_power += power

        final_score = weighted_sum / total_power if total_power > 0 else 0
        final_decision = 1 if final_score >= 0.5 else 0

        # 4. 結果のフィードバックと学習 (Social Learning)
        is_correct = None
        if ground_truth is not None:
            is_correct = (final_decision == ground_truth)

            for agent_id in self.agents.keys():
                # ToMの更新
                target_id = delegation_map.get(
                    agent_id, agent_id)  # 委任してなければ自分

                # 相手の個別の判断が正しかったか？
                target_dec, _ = initial_decisions[target_id]
                target_correct = (target_dec == ground_truth)

                # ToMのモデル更新
                outcome_val = 1.0 if target_correct else 0.0
                self.toms[agent_id].update_model(target_id, outcome_val)

        return {
            'consensus_decision': final_decision,
            'vote_ratio': final_score,
            'delegation_count': len(delegation_map),
            'correct': is_correct
        }
