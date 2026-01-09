# ファイルパス: scripts/experiments/systems/run_collective_intelligence.py
# 日本語タイトル: Collective Intelligence Simulation (LDP) - Type Fixed
# 修正内容: ArchitectureRegistry.build の戻り値を SpikingWorldModel にキャスト。

import os
import sys
import torch
import logging
from typing import cast, Tuple

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())

from snn_research.core.architecture_registry import ArchitectureRegistry
from snn_research.models.experimental.brain_v4 import SynestheticBrain
from snn_research.models.experimental.world_model_snn import SpikingWorldModel
from snn_research.agent.synesthetic_agent import SynestheticAgent
from snn_research.social.theory_of_mind import TheoryOfMindModule
from snn_research.collective.liquid_democracy import LiquidDemocracyProtocol

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CollectiveSim")

def create_agent(name: str, device: str, noise_level: float = 0.0) -> Tuple[SynestheticAgent, TheoryOfMindModule, float]:
    """
    エージェントと、そのToMモジュールを生成するファクトリ関数。
    noise_levelが高いエージェントは判断を誤りやすい（＝素人）。
    """
    # 1. Brain & WM (Small config for simulation)
    brain = SynestheticBrain(
        vocab_size=100, d_model=32, num_layers=1, time_steps=4, device=device
    )
    wm_config = {
        'd_model': 32, 'd_state': 16, 'num_layers': 1, 'time_steps': 4, 
        'action_dim': 1, 'sensory_configs': {'vision': 32}
    }
    # mypy fix: Explicit cast to SpikingWorldModel
    wm_module = ArchitectureRegistry.build("spiking_world_model", wm_config, 0).to(device)
    wm = cast(SpikingWorldModel, wm_module)
    
    # 2. Agent Wrapper
    agent = SynestheticAgent(brain, wm, action_dim=1, device=device)
    
    # 3. Theory of Mind
    tom = TheoryOfMindModule(observation_dim=4, hidden_dim=16, history_len=5).to(device)
    
    return agent, tom, noise_level

def generate_task_data(batch_size: int, input_dim: int, device: str):
    """
    タスク: ランダムなベクトルを入力し、その「平均値が正か負か」を当てるバイナリ分類。
    """
    data = torch.randn(batch_size, input_dim, device=device)
    # 正解: 平均が0より大きければ1, それ以外は0
    labels = (data.mean(dim=1) > 0).long()
    return data, labels

def main():
    logger.info("🐝 Starting Collective Intelligence Simulation (Liquid Democracy)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_rounds = 30
    input_dim = 32
    
    # --- 1. Agent Population Setup ---
    # 5体のエージェントを作成。能力(ノイズ耐性)に差をつける。
    agent_configs = [
        ("Expert_A", 0.0),   # 非常に優秀 (Noise 0)
        ("Average_B", 0.5),  # 普通
        ("Average_C", 0.5),
        ("Novice_D", 1.0),   # 判断がランダムに近い
        ("Novice_E", 1.0)
    ]
    
    agents = {}
    toms = {}
    noise_profiles = {}
    
    for name, noise in agent_configs:
        agent, tom, n_level = create_agent(name, device, noise)
        agents[name] = agent
        toms[name] = tom
        noise_profiles[name] = n_level
        logger.info(f"   - Created Agent: {name} (Noise Level: {n_level})")

    # --- 2. Initialize Protocol ---
    ldp = LiquidDemocracyProtocol(agents, toms)
    
    history_accuracy = []
    delegation_counts = []
    
    # --- 3. Simulation Loop ---
    for round_idx in range(num_rounds):
        # タスク生成
        task_input, label = generate_task_data(1, input_dim, device)
        task_input = task_input[0] # (D,)
        ground_truth = label[0].item()
        
        result = ldp.conduct_vote(task_input, ground_truth)
        
        # ログ
        acc = 1.0 if result['correct'] else 0.0
        history_accuracy.append(acc)
        delegation_counts.append(result['delegation_count'])
        
        logger.info(f"Round {round_idx+1}: Consensus={result['consensus_decision']} (Truth={ground_truth}) "
                    f"| Delegations={result['delegation_count']} | Correct={result['correct']}")

    # --- 4. Result Analysis ---
    avg_acc = sum(history_accuracy) / len(history_accuracy)
    avg_del = sum(delegation_counts) / len(delegation_counts)
    
    logger.info("\n📊 Simulation Result:")
    logger.info(f"   Average Accuracy: {avg_acc:.2%}")
    logger.info(f"   Avg Delegation Count: {avg_del:.1f} / {len(agents)}")
    
    if avg_acc > 0.6:
        logger.info("✅ Collective Intelligence Emerged! The group performed better than random.")
    else:
        logger.info("⚠️ Performance Low. Agents need more training to trust experts.")

if __name__ == "__main__":
    main()