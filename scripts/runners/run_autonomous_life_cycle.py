# ファイルパス: scripts/runners/run_autonomous_life_cycle.py
# 日本語タイトル: Autonomous Life Cycle Runner (Day/Night Simulation)
# 目的・内容:
#   ROADMAP Phase 2 完成形。
#   身体(Agent)、心(Motivator)、脳(Sleep)、生理(Homeostasis)を統合し、
#   昼夜のサイクルを通じて自律的に成長し続けるAIのライフサイクルを実行する。

import os
import sys
import torch
import torch.nn as nn
import logging
import time
import random
from typing import Tuple

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# ログ設定: 重要なイベントだけ見やすく表示
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

# Modules
from snn_research.core.architecture_registry import ArchitectureRegistry
from snn_research.systems.embodied_vlm_agent import EmbodiedVLMAgent
from snn_research.adaptive.intrinsic_motivator import IntrinsicMotivator
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.cognitive_architecture.homeostasis import Homeostasis
from snn_research.systems.autonomous_learning_loop import AutonomousLearningLoop

# --- Dummy Environment ---
class VirtualWorld:
    """エージェントが探索する仮想環境"""
    def __init__(self, img_size=32, device="cpu"):
        self.device = device
        self.img_size = img_size
        self.current_scene = torch.randn(1, 3, img_size, img_size).to(device)
        self.time_of_day = 0.0 # 0.0(Morning) -> 1.0(Night)

    def observe(self):
        return self.current_scene

    def step(self, action: torch.Tensor):
        # Action changes the scene slightly
        change = torch.randn_like(self.current_scene) * 0.1 * action.mean().item()
        self.current_scene = torch.clamp(self.current_scene + change, -1, 1)
        
        # Time passes
        self.time_of_day += 0.05
        return self.current_scene

# --- Main Simulation ---
def run_life_cycle():
    print("""
    =======================================================
       🧬 SNN AUTONOMOUS LIFE CYCLE SIMULATION v1.0 🧬
       
       Components:
       - Body: Embodied VLM (Vision-Language-Motor)
       - Mind: Intrinsic Motivation (Curiosity)
       - Brain: Hippocampal Replay (Sleep & Dream)
       - Life: Homeostasis (Fatigue & Energy)
    =======================================================
    """)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"⚙️ System initializing on {device}...")
    
    # 1. Config & Model Build
    vocab_size = 1000
    img_size = 32
    
    full_config = {
        "vision_config": { "type": "cnn", "hidden_dim": 64, "img_size": img_size, "time_steps": 4, "neuron": {"type": "lif"} },
        "language_config": { "d_model": 64, "vocab_size": vocab_size, "num_layers": 2, "time_steps": 4 },
        "projector_config": {"projection_dim": 64},
        "sensory_inputs": {"vision": 64},
        "use_bitnet": False
    }
    motor_config = {"action_dim": 2, "hidden_dim": 32}

    # Agent Construction
    try:
        vlm_model = ArchitectureRegistry.build("spiking_vlm", full_config, vocab_size)
    except:
        from snn_research.models.transformer.spiking_vlm import SpikingVLM
        vlm_model = SpikingVLM(vocab_size, full_config["vision_config"], full_config["language_config"], projection_dim=64)

    agent = EmbodiedVLMAgent(vlm_model, motor_config).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3)
    
    # Subsystems
    homeostasis = Homeostasis(config={"fatigue_rate": 2.0, "sleep_threshold": 20.0}).to(device) # Fast fatigue for demo
    sleeper = SleepConsolidator(agent, optimizer, buffer_size=100, batch_size=4, device=device)
    learning_loop = AutonomousLearningLoop(agent, optimizer, device=device)
    
    world = VirtualWorld(img_size=img_size, device=device)
    
    # Simulation Parameters
    days_to_simulate = 2
    context_text = torch.randint(0, vocab_size, (1, 8)).to(device)
    
    # --- Life Cycle Loop ---
    for day in range(1, days_to_simulate + 1):
        homeostasis.new_day()
        world.time_of_day = 0.0
        
        logger.info(f"☀️ --- DAY {day} START ---")
        
        # Day Phase: Activity & Exploration
        while True:
            # 1. Check Body Status
            status = homeostasis.check_needs()
            stats = homeostasis.get_status()
            
            if status == "sleep":
                logger.info(f"🥱 Feeling sleepy... (Fatigue: {stats['fatigue']:.1f}%)")
                break # Go to night phase
            
            # 2. Perceive & Act (Autonomous Step)
            current_obs = world.observe()
            
            # Agent decides action
            with torch.no_grad():
                # For action generation (Behavior)
                agent_out = agent(current_obs, context_text)
                action = agent_out["action_pred"]
            
            # 3. Environment Response
            next_obs = world.step(action)
            
            # 4. Learning (Curiosity)
            # 予測誤差を計算し、世界モデルを更新
            metrics = learning_loop.step(current_obs, context_text, next_obs)
            reward = metrics["intrinsic_reward"]
            
            # 5. Memory Formation (Hippocampus)
            # 経験を短期記憶に保存（睡眠時のリプレイ用）
            sleeper.store_experience(current_obs, context_text, reward)
            
            # 6. Update Physiology
            # 行動強度に応じて疲労蓄積
            action_intensity = action.abs().mean().item() * 10.0
            homeostasis.update(action_intensity + 1.0) # Base metabolism
            
            # Logging (sparse)
            if random.random() < 0.2:
                logger.info(f"   🏃 Acting... | Curiosity Reward: {reward:.4f} | Fatigue: {stats['fatigue']:.1f}%")
                
            # Avoid infinite loop in case of balance issues
            if world.time_of_day > 2.0: 
                logger.warning("   🕰️ It's getting too late. Forcing sleep.")
                break

        # Night Phase: Sleep & Consolidation
        logger.info(f"🌙 --- NIGHT {day} (Dreaming Phase) ---")
        
        # 1. Sleep Consolidation (Replay Learning)
        # 日中に溜めた記憶から、重要なもの（Reward高）をリプレイ
        dream_stats = sleeper.sleep(cycles=5)
        
        # 2. Physiological Recovery
        while homeostasis.fatigue > 0:
            homeostasis.rest()
            
        logger.info(f"   💤 Dream Loss: {dream_stats['sleep_loss']:.4f}")
        logger.info(f"   🔋 Fully recovered. Ready for tomorrow.")
        
    logger.info("🎉 Life Cycle Simulation Completed Successfully.")

if __name__ == "__main__":
    run_life_cycle()