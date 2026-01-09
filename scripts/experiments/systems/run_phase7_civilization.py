# ファイルパス: scripts/experiments/systems/run_phase7_civilization.py
# 日本語タイトル: Phase 7 Civilization Simulation - Multi-Agent Consensus
# 目的: 複数のArtificialBrainエージェントによる社会形成、合意形成、知識継承のシミュレーション。

from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.social.culture_repository import CultureRepository
from snn_research.social.consensus_engine import ConsensusEngine
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
import asyncio
import logging
import torch
import sys
import os
import random
from typing import Optional

# パス設定
sys.path.append(os.getcwd())


# ロギング設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase7_Civ")


class SocialAgent:
    """
    社会シミュレーションに参加するエージェントのラッパー。
    ArtificialBrainを持ち、社会的な対話インターフェースを提供する。
    """

    def __init__(self, id: int, device: str):
        self.id = id
        self.name = f"Agent_{id:02d}"

        # エージェントごとの個性を出すため、シードを少しずらすなどの工夫が可能
        # ここでは軽量化のため最小構成のBrainを使用
        self.brain = ArtificialBrain(
            global_workspace=GlobalWorkspace(),
            spike_encoder=SpikeEncoder(device=device),
            device=device
        )

        self.confidence = 0.5
        # mypyエラー修正: 型ヒントを追加し、TensorまたはNoneであることを明示
        self.proposal: Optional[torch.Tensor] = None
        self.device = device

    async def think(self, topic: str):
        """トピックについて思考し、提案ベクトルと自信度を生成する"""
        logger.info(f"🤖 {self.name} is thinking about '{topic}'...")

        # 1. 思考シミュレーション (Brainを実行)
        # 実際の思考回路を通すが、出力はテキストやアクションになることが多い
        _ = self.brain.run_cognitive_cycle(topic)

        # 2. 提案ベクトルの生成 (Simulation)
        # 本来はBrainの内部状態（SNNの隠れ層など）からベクトルを抽出するが、
        # ここではシミュレーションとして、IDに基づくバイアスを加えたベクトルを生成する。
        # トピックの意味ベクトルに近いものを目指すが、各個体でズレがある状態を再現。

        # 簡易的なトピックベクトル（正解のようなもの）
        topic_hash = abs(hash(topic)) % 1000 / 1000.0
        target_vec = torch.ones(16, device=self.device) * topic_hash

        # 個体のバイアス (Noise)
        bias = torch.randn(16, device=self.device) * 0.2
        # エージェントIDによる特有の傾向 (Personality)
        personality = torch.tensor([self.id * 0.05] * 16, device=self.device)

        self.proposal = target_vec + bias + personality

        # 3. 自信度の生成
        # 提案ベクトルがどれだけ強固か（ここではランダム要素 + 経験値）
        self.confidence = max(0.1, min(0.9, 0.5 + random.uniform(-0.2, 0.2)))

        return self.proposal, self.confidence


async def main():
    logger.info("==================================================")
    logger.info("   🌍 Phase 7 Civilization Simulation Start")
    logger.info("==================================================")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running on device: {device}")

    # 1. 社会インフラの構築
    num_agents = 3
    # 合意形成エンジン
    consensus_engine = ConsensusEngine(
        num_agents=num_agents, proposal_dim=16, device=device)
    # 文化リポジトリ
    culture_repo = CultureRepository()

    # エージェントの生成
    agents = [SocialAgent(i, device) for i in range(num_agents)]
    logger.info(f"Population: {len(agents)} agents created.")

    # 2. シミュレーション: 問題解決と合意形成
    topic = "Optimal resource allocation strategy for sustainability"
    logger.info(f"\n--- 🗣️ Debate Topic: {topic} ---")

    # 全エージェントが思考し、提案を提出
    proposals = []
    confidences = []

    for agent in agents:
        p, c = await agent.think(topic)
        proposals.append(p)
        confidences.append(c)
        logger.info(f"   - {agent.name}: Confidence={c:.2f}")

    # Tensor化 (Batch化)
    proposals_tensor = torch.stack(proposals)  # [Num_Agents, 16]
    confidences_tensor = torch.tensor(
        confidences, device=device).unsqueeze(1)  # [Num_Agents, 1]

    # 3. 合意形成エンジンの実行 (リキッドデモクラシー)
    logger.info("\n⚖️ Running Consensus Engine (Liquid Democracy)...")
    consensus_result = consensus_engine(proposals_tensor, confidences_tensor)

    status = consensus_result['status']
    coherence = consensus_result['coherence']
    effective_power = consensus_result['effective_power']

    logger.info(f"Consensus Status: {status}")
    logger.info(f"Coherence Score: {coherence:.4f}")
    logger.info("Effective Power Distribution (Voting Influence):")
    for i, power in enumerate(effective_power):
        logger.info(f"  - {agents[i].name}: {power:.4f}")

    # 4. 文化の継承と報酬
    if status == "AGREED":
        logger.info(
            "\n--- 📜 Consensus Reached: Recording to Culture Repository ---")

        # 最も影響力のあった（貢献した）エージェントを特定
        best_agent_idx = effective_power.argmax().item()
        best_agent = agents[best_agent_idx]

        # 合意ベクトルを文化として保存
        culture_repo.contribute_meme(
            concept_name="Sustainability_Strategy_v1",
            vector=consensus_result['consensus_vector'],
            description=f"Consensus reached on {topic} by {num_agents} agents.",
            utility_score=coherence
        )

        # 社会的報酬の付与 (信頼度マトリクスの更新)
        # 貢献者には大きな報酬、他者にも参加報酬
        rewards = [0.1] * num_agents
        rewards[best_agent_idx] = 1.0  # Winner takes more trust

        consensus_engine.update_trust(list(range(num_agents)), rewards)
        logger.info(
            f"Trust updated. {best_agent.name} gained significant reputation.")

    else:
        logger.warning(
            "\n❌ Consensus failed (DISPUTED). No culture recorded. Further debate needed.")

    # 5. 知識の検索テスト (次世代への継承シミュレーション)
    logger.info("\n--- 🔍 Future Generation Learning ---")
    knowledge = culture_repo.retrieve_meme("Sustainability_Strategy_v1")

    if knowledge:
        logger.info("New Agent retrieved knowledge from history:")
        logger.info(f"   Name: {knowledge['name']}")
        logger.info(f"   Generation: {knowledge['generation']}")
        logger.info(f"   Utility: {knowledge['utility']:.4f}")
        logger.info(
            "   -> This meme is now part of the collective unconscious.")

    logger.info("\n✅ Civilization Simulation Completed.")

if __name__ == "__main__":
    asyncio.run(main())
