# ファイルパス: snn_research/collective/knowledge_exchange.py
# Title: Knowledge Exchanger v1.0
# Description:
#   エージェント間で知識（重み、概念、経験）を共有するためのモジュール。
#   Federated Learning的な重み平均化と、概念アライメントをサポートする。

import torch
import torch.nn as nn
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class KnowledgeExchanger:
    """
    集団知識交換プロトコルハンドラ。
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        logger.info(f"📚 KnowledgeExchanger initialized for Agent {agent_id}.")

    def aggregate_weights(self, my_model: nn.Module, peer_weights: List[Dict[str, torch.Tensor]], alpha: float = 0.5):
        """
        他者の重みを取り入れ、自分のモデルを更新する (Federated Averaging-like)。
        alpha: 自分の重みの保持率 (0.0 - 1.0)
        """
        if not peer_weights:
            return

        with torch.no_grad():
            my_state = my_model.state_dict()
            avg_peer_state = {}

            # ピアの重みの平均計算
            num_peers = len(peer_weights)
            first_peer = peer_weights[0]

            for key in first_peer.keys():
                if key in my_state:  # 自分のモデルにあるキーのみ
                    sum_tensor = torch.zeros_like(my_state[key])
                    count = 0
                    for peer in peer_weights:
                        if key in peer:
                            sum_tensor += peer[key]
                            count += 1

                    if count > 0:
                        avg_peer_state[key] = sum_tensor / count

            # 統合: New = alpha * Old + (1 - alpha) * PeerAvg
            for key, peer_tensor in avg_peer_state.items():
                my_state[key] = alpha * my_state[key] + \
                    (1.0 - alpha) * peer_tensor

            my_model.load_state_dict(my_state)
            logger.info(
                f"   🔄 Merged knowledge from {num_peers} peers (Alpha: {alpha}).")

    def create_concept_packet(self, concept_id: str, centroid: torch.Tensor, description: str) -> Dict[str, Any]:
        """
        概念共有用のパケットを作成する。
        """
        return {
            "type": "concept_share",
            "sender": self.agent_id,
            "concept_id": concept_id,
            "centroid": centroid.tolist(),  # JSON serializable
            "description": description
        }
