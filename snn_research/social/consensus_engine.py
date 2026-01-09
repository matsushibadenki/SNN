# ファイルパス: snn_research/social/consensus_engine.py
# Title: Spiking Consensus Engine (Liquid Democracy Core)
# Description:
# - Phase 7: 複数エージェント間の合意形成を行うモジュール。
# - Objective 21: リキッドデモクラシー（流動的民主主義）をSNNの興奮・抑制ダイナミクスでモデル化。
# - 信頼度（Synaptic Weight）に基づく投票権の委譲と、集団的意思決定を実現する。

import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ConsensusEngine(nn.Module):
    """
    スパイキング合意形成エンジン。
    複数のエージェント(ニューロン集団)からの入力を統合し、
    信頼重み付けを行った上で集団としての決定を下す。
    """
    def __init__(self, num_agents: int, proposal_dim: int = 16, device: str = 'cpu'):
        super().__init__()
        self.num_agents = num_agents
        self.proposal_dim = proposal_dim # 提案のベクトル次元
        self.device = device
        
        # 信頼マトリクス (Trust Matrix): Agent i が Agent j をどれだけ信頼しているか
        # 初期値は自己信頼のみ高く、他者はフラット。学習により動的に変化する。
        # 行i: Agent iからの信頼配分
        self.trust_matrix = nn.Parameter(torch.eye(num_agents) * 0.5 + 0.1)
        
        self.to(device)
        logger.info(f"⚖️ Consensus Engine initialized for {num_agents} agents.")

    def forward(self, agent_proposals: torch.Tensor, agent_confidences: torch.Tensor) -> Dict[str, Any]:
        """
        集団的合意形成を実行する。

        Args:
            agent_proposals: 各エージェントの提案ベクトル [Num_Agents, Proposal_Dim]
            agent_confidences: 各エージェントの自信度 [Num_Agents, 1]
        
        Returns:
            Dict: 合意結果ベクトル、各エージェントの実効影響力、合意形成の強度(Coherence)
        """
        # 1. 流動的委譲 (Liquid Delegation)
        # 信頼マトリクスを用いて、自信のないエージェントの「票（重み）」を信頼できるエージェントに流す
        # Trust[i, j] * Confidence[j] -> iの票がjの影響力を強化
        
        with torch.no_grad():
            # 信頼度と自信度の積（信頼している相手が自信を持っている場合に強く委譲）
            weighted_trust = self.trust_matrix * agent_confidences.T
            
            # 行ごとに正規化（各エージェントの票の配分比率）
            delegation_flow = torch.nn.functional.softmax(weighted_trust, dim=1)
            
            # 各エージェントの実効影響力 (Effective Power)
            # 全員からの委譲の総和 (列方向の和)
            effective_power = delegation_flow.sum(dim=0) # [Num_Agents]
            effective_power = effective_power / effective_power.sum() # 正規化して総和を1にする

        # 2. 加重投票 (Weighted Voting)
        # エージェントの提案ベクトルを、実効影響力で重み付けして統合（平均）
        # [Num_Agents, Dim] * [Num_Agents, 1] (broadcast) -> [Num_Agents, Dim] -> Sum -> [Dim]
        
        weighted_proposals = agent_proposals * effective_power.unsqueeze(1)
        consensus_vector = weighted_proposals.sum(dim=0)
        
        # 3. 合意強度の判定 (Coherence)
        # 全員の提案がどれくらい一致しているか（ベクトルの分散の逆数的な指標）
        # 分散が小さいほど、合意強度は高い
        variance = torch.var(agent_proposals, dim=0).mean().item()
        coherence = 1.0 / (1.0 + variance * 10.0) # スケーリング
        
        status = "AGREED" if coherence > 0.5 else "DISPUTED"

        return {
            "consensus_vector": consensus_vector,
            "effective_power": effective_power,
            "coherence": coherence,
            "status": status
        }

    def update_trust(self, agent_indices: List[int], rewards: List[float]):
        """
        結果に基づいて信頼マトリクスを更新する (Social Learning)。
        良い提案をした（報酬を得た）エージェントへの信頼を社会全体で高める。
        """
        with torch.no_grad():
            for idx, reward in zip(agent_indices, rewards):
                if reward == 0:
                    continue
                    
                # 報酬が正なら、そのエージェントに対する他者(全員)からの信頼を少し上げる
                # 行k (他人) -> 列idx (対象エージェント) の値を更新
                if reward > 0:
                    self.trust_matrix[:, idx] += 0.05 * reward
                else:
                    self.trust_matrix[:, idx] -= 0.05 * abs(reward)
            
            # 値を0-1にクリップし、自身の自己信頼(対角成分)が極端に低くならないように調整
            self.trust_matrix.data.clamp_(0.0, 1.0)
            # 行ごとの再正規化は次のforwardで行われるため、ここではパラメータ更新のみ