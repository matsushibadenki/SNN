# Title: SNN Project Unified Mission Runner "The Odyssey"
# Description:
#   Phase 2 (Sleep), Phase 3 (Embodiment), Phase 4 (Collective) を統合した最終デモ。
#   シナリオ: 未知の惑星探査ミッション。
#   1. 高速移動 (Reflex/Embodiment)
#   2. 未知物体の解析 (Collective Intelligence)
#   3. 休息と学習 (Sleep & Consolidation)

import sys
import os
import time
import logging
import torch
import torch.nn as nn
from typing import Dict, Any

# --- 環境設定 & ロギング ---
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("UnifiedMission")

# --- Import Core Components ---
try:
    from snn_research.core.neurons.da_lif_node import DualAdaptiveLIFNode
    from snn_research.io.spike_encoder import HybridTemporal8BitEncoder
    from snn_research.models.transformer.spikformer import Spikformer, TransformerToMambaAdapter
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
    from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
    from snn_research.collective.liquid_democracy import LiquidDemocracyProtocol, Proposal
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

# --- Integrated Agent Class ---


class IntegratedBrainAgent(nn.Module):
    """
    Brain v22: The Collective Mind Embodied
    身体性(v21)と社会性(v22)を併せ持つ統合エージェント。
    """

    def __init__(self, agent_id: str, device: str):
        super().__init__()
        self.agent_id = agent_id
        self.device = device
        self.role = "Explorer"

        logger.info(
            f"🧠 Initializing Agent '{agent_id}' on {device.upper()}...")

        # 1. Neural Engine (Reflex & Perception) - T=1 for Speed
        self.encoder = HybridTemporal8BitEncoder(duration=1)

        # 軽量化された視覚野 (128 dim)
        self.visual_cortex = Spikformer(
            img_size_h=128, img_size_w=128, patch_size=16,
            embed_dim=128, num_heads=4, num_layers=2, T=1
        ).to(device)

        # 思考回路 (PFC)
        self.adapter = TransformerToMambaAdapter(
            vis_dim=128, model_dim=256, seq_len=64).to(device)
        self.pfc = BitSpikeMamba(
            vocab_size=1000, d_model=256, d_state=32, d_conv=4, expand=2,
            num_layers=2, time_steps=1,
            neuron_config={"type": "lif",
                           "tau_mem": 2.0, "base_threshold": 1.0}
        ).to(device)

        # 反射モジュール
        self.reflex = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(64*128, 32),
            DualAdaptiveLIFNode(detach_reset=True),
            nn.Linear(32, 5)  # 5 actions
        ).to(device)

        # 2. Physiological Engine (Energy)
        self.astrocyte = AstrocyteNetwork(max_energy=500.0)

        # 3. Social Engine (State)
        self.confidence_level = 1.0
        self.reputation = 1.0

    def perceive_and_act(self, visual_input: torch.Tensor) -> Dict[str, Any]:
        """Phase 1: 高速反射動作"""
        if not self.astrocyte.request_resource("cortex", 2.0):
            return {"action": "REST", "status": "Low Battery"}

        # Encoding & Vision
        spikes = self.encoder(visual_input, duration=1)
        target_dtype = self.visual_cortex.patch_embed.weight.dtype
        if spikes.dtype != target_dtype:
            spikes = spikes.to(dtype=target_dtype)

        features = self.visual_cortex(spikes)

        # Reflex
        reflex_out = self.reflex(features)
        action_idx = reflex_out.mean(dim=1).argmax(dim=-1).item()

        # Confidence Simulation (Entropy based)
        probs = torch.softmax(reflex_out.mean(dim=1), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-6)).sum().item()
        self.confidence_level = max(0.0, 1.0 - (entropy * 0.5))

        return {
            "action": action_idx,
            "confidence": self.confidence_level,
            "energy": self.astrocyte.get_energy_level()
        }

    def sleep_and_consolidate(self):
        """Phase 3: 睡眠と記憶の定着"""
        self.astrocyte.clear_fatigue(50.0)
        self.astrocyte.replenish_energy(200.0)
        # 実際にはここで重みの更新（Replay）を行う
        logger.info(
            f"💤 {self.agent_id} is sleeping... (Consolidating Memories)")
        time.sleep(1.0)
        logger.info(f"🌅 {self.agent_id} woke up! Energy restored.")

# --- Simulation Utilities ---


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MissionControl:
    def __init__(self):
        self.protocol = LiquidDemocracyProtocol()
        # 専門家エージェントを登録しておく
        self.protocol.reputations["Mission_Commander"] = 5.0  # 絶対的信頼
        self.protocol.reputations["Xeno_Biologist"] = 3.0    # 生物学の専門家

    def request_swarm_consensus(self, agent: IntegratedBrainAgent, task_desc: str):
        logger.info(
            f"📡 [Swarm] Distress signal from {agent.agent_id}: '{task_desc}'")

        # 提案の生成 (本来は各エージェントが出すが、ここではシミュレート)
        proposals = [
            Proposal(id="P1", description="Approach and Scan (Safe)",
                     content="SCAN"),
            Proposal(id="P2", description="Attack (Hostile)", content="ATTACK")
        ]

        # 投票フェーズ
        # エージェント本人（自信なし）
        self.protocol.cast_vote(agent.agent_id, "", False,
                                0.0, delegate_to="Xeno_Biologist")

        # 専門家（自信あり）
        self.protocol.cast_vote("Xeno_Biologist", "P1", True, 0.9)
        self.protocol.cast_vote("Mission_Commander", "P1", True, 0.8)

        # 集計
        scores = self.protocol.tally_votes(
            proposals, self.protocol.vote_history)
        winner_id = max(scores, key=scores.get)
        winner = next(p for p in proposals if p.id == winner_id)

        return winner

# --- Main Mission Scenario ---


def run_mission():
    print("\n" + "="*60)
    print("   SNN PROJECT: UNIFIED MISSION DEMO 'THE ODYSSEY'   ")
    print("="*60 + "\n")

    device = get_optimal_device()
    agent = IntegratedBrainAgent("Agent_Odyssey", device)
    mission_ctrl = MissionControl()

    # --- PHASE 1: Solo Exploration (Reflex) ---
    logger.info(">>> PHASE 1: Planetary Exploration (High-Speed Reflex)")
    logger.info(
        "   Target: Navigate through asteroid field. Latency must be < 30ms.")

    # ウォームアップ
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    for _ in range(5):
        agent.perceive_and_act(dummy_input)

    # 走行デモ

    steps = 50
    total_latency = 0

    with torch.no_grad():
        for i in range(steps):
            step_start = time.perf_counter()

            # 視覚入力（ランダム生成）
            obs = torch.randn(1, 3, 128, 128, device=device)
            result = agent.perceive_and_act(obs)

            latency = (time.perf_counter() - step_start) * 1000
            total_latency += latency

            # 疲労蓄積
            agent.astrocyte.log_fatigue(0.02)

            if i % 10 == 0:
                logger.info(
                    f"   Step {i:02d}: Action={result['action']} | Latency={latency:.2f}ms | Energy={result['energy']:.2f}")

    avg_latency = total_latency / steps
    logger.info(f"✅ Phase 1 Complete. Avg Latency: {avg_latency:.2f}ms")

    # --- PHASE 2: Unknown Encounter (Collective) ---
    logger.info("\n>>> PHASE 2: Anomaly Detection (Collective Intelligence)")
    logger.info(
        "   Alert: Unknown biological signature detected. Confidence dropping.")

    # 自信レベルを強制的に下げる（未知の物体）
    agent.confidence_level = 0.2
    logger.info(
        f"   ⚠️ {agent.agent_id} Confidence: {agent.confidence_level:.2f} (Too low to act)")

    # スワームへ支援要請
    decision = mission_ctrl.request_swarm_consensus(
        agent, "Analyze Unknown Creature")

    logger.info(f"   🗳️ Swarm Decision: {decision.description}")

    if decision.content == "SCAN":
        logger.info(
            "   🤖 Action Executed: SCAN initiated. Creature is friendly.")
        # 成功したので報酬を得る（評判アップ）
        mission_ctrl.protocol.update_reputation(decision.id, 1.0)

    # --- PHASE 3: Adaptation (Sleep) ---
    logger.info("\n>>> PHASE 3: Night Cycle (Sleep & Consolidation)")

    # 現在のステータス
    stats = agent.astrocyte.get_diagnosis_report()
    logger.info(
        f"   Status before sleep: Fatigue={stats['metrics']['fatigue_level']:.2f}")

    # 睡眠実行
    agent.sleep_and_consolidate()

    stats_after = agent.astrocyte.get_diagnosis_report()
    logger.info(
        f"   Status after sleep:  Fatigue={stats_after['metrics']['fatigue_level']:.2f}")

    logger.info("\n" + "="*60)
    logger.info("   MISSION ACCOMPLISHED")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    try:
        run_mission()
    except KeyboardInterrupt:
        logger.info("Mission aborted.")
    except Exception as e:
        logger.error(f"Mission failed: {e}")
        import traceback
        traceback.print_exc()
