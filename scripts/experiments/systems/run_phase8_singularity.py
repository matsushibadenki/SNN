# ファイルパス: scripts/experiments/systems/run_phase8_singularity.py
# Title: Phase 8 Singularity Event Simulation (Verbose Mode)
# Description:
# - ロギング設定を強化し、進行状況を詳細に表示するよう修正。
# - 初期化ステップごとにStatusを表示。

import asyncio
import logging
import torch
import sys
import os

# パス設定
sys.path.append(os.getcwd())

from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.thalamus import Thalamus
from snn_research.core.neuromorphic_os import NeuromorphicOS
from snn_research.core.omega_point import OmegaPointSystem
from snn_research.io.spike_encoder import SpikeEncoder

# ロギング設定 (強制的に標準出力へ)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger("Singularity_Sim")

async def main():
    print("\n" + "="*50)
    print("   ♾️  Phase 8: The Omega Point Simulation")
    print("="*50 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Hardware Acceleration: {device.upper()}")

    # 1. シードとなる人工脳の構築 (Seed AI)
    logger.info("🌱 Constructing Seed AI (Gen 0)...")
    
    try:
        print("   -> Initializing Global Workspace...")
        workspace = GlobalWorkspace()
        
        print("   -> Initializing Thalamus...")
        thalamus = Thalamus(device=device)
        
        print("   -> Initializing Spike Encoder...")
        encoder = SpikeEncoder(device=device)
        
        print("   -> Assembling Artificial Brain (This may take a moment)...")
        seed_brain = ArtificialBrain(
            global_workspace=workspace,
            thalamus=thalamus,
            spike_encoder=encoder,
            device=device
        )
        logger.info("✅ Seed Brain constructed successfully.")

    except Exception as e:
        logger.error(f"❌ Failed to construct Seed AI: {e}")
        return
    
    # OSの初期化
    print("   -> Booting Neuromorphic OS Kernel...")
    os_kernel = NeuromorphicOS(seed_brain)
    
    # 2. オメガ・ポイント・システムの起動
    print("   -> Initializing Omega Point Control System...")
    omega_system = OmegaPointSystem(seed_brain, os_kernel)
    
    # 3. シンギュラリティ・イベントの実行
    logger.info("\n--- 🌀 Starting Recursive Self-Improvement ---")
    print("   (Note: Simulation runs an evolutionary loop. Please wait...)\n")
    
    try:
        await omega_system.ignite_singularity(target_metric_score=95.0)
    except Exception as e:
        logger.error(f"❌ Error during singularity loop: {e}", exc_info=True)
        return
    
    # 4. 最終形態の確認
    logger.info("\n--- ✨ Post-Singularity Analysis ---")
    final_brain = omega_system.brain
    status = final_brain.get_brain_status()
    
    logger.info(f"Final Brain Status: {status['status']}")
    logger.info(f"Evolution Generations: {omega_system.improver.generation}")
    
    # 最終的な脳でタスクを実行してみる
    logger.info("Executing task with Evolved Brain...")
    test_input = "What is the meaning of existence?"
    try:
        result = final_brain.run_cognitive_cycle(test_input)
        logger.info(f"Response: {result.get('response')}")
    except Exception as e:
        logger.error(f"Failed to execute final task: {e}")
    
    logger.info("✅ Singularity Simulation Completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())