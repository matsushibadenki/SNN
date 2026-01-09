# ファイルパス: scripts/experiments/social/run_synesthetic_communication.py
# 日本語タイトル: Run Synesthetic Communication Experiment
# 目的: 2つのエージェント(Alice, Bob)による画像描写ゲームを実行し、
#       言語を通じたイメージ共有（記号接地）の成立過程をシミュレーションする。

import os
import sys
import torch
import logging
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())

from snn_research.core.architecture_registry import ArchitectureRegistry
from snn_research.models.experimental.brain_v4 import SynestheticBrain
from snn_research.agent.synesthetic_agent import SynestheticAgent
from snn_research.social.communication_channel import CommunicationChannel
from snn_research.social.synesthetic_dialogue import SynestheticDialogue

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SocialSim")

def generate_random_visual_concept(batch_size: int, feat_dim: int, device: str) -> torch.Tensor:
    """
    ランダムな「視覚概念」を生成する。
    例: 「赤い丸」に相当する特徴ベクトルなど。
    """
    # 実際には画像データセット(CIFAR/MNIST)を使うのが良いが、
    # ここでは特徴空間上のランダムベクトルで代用
    return torch.randn(batch_size, 1, feat_dim, device=device) # (B, 1, D)

def main():
    logger.info("🗣️ Starting Synesthetic Communication Experiment...")
    
    # --- 1. Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        'vocab_size': 100, # 小規模な語彙で実験
        'd_model': 64,
        'vision_dim': 64,  # Brainのd_modelと合わせる(簡易化)
        'noise_level': 0.05
    }
    
    # --- 2. Build Agents (Alice & Bob) ---
    logger.info("   - Creating Agents: Alice (Speaker) & Bob (Listener)...")
    
    def create_agent(name):
        brain = SynestheticBrain(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_layers=2,
            time_steps=4,
            device=device
        )
        # World Model (Decoderとして使用)
        wm_config = {
            'd_model': config['d_model'],
            'd_state': 32, 'num_layers': 2, 'time_steps': 4, 'action_dim': 2,
            'sensory_configs': {'vision': config['vision_dim']} # Vision Decoderを持つ
        }
        wm = ArchitectureRegistry.build("spiking_world_model", wm_config, 0).to(device)
        return SynestheticAgent(brain, wm, action_dim=2, device=device)

    alice = create_agent("Alice")
    bob = create_agent("Bob")
    
    # --- 3. Setup Environment ---
    channel = CommunicationChannel(noise_level=config['noise_level'], device=device)
    dialogue = SynestheticDialogue(alice, bob, channel, vocab_size=config['vocab_size'])
    
    # --- 4. Simulation Loop ---
    num_rounds = 20
    history_similarity = []
    
    logger.info(f"   - Running {num_rounds} communication rounds...")
    
    for round_idx in range(num_rounds):
        # A. 共通の「お題」画像 (Visual Concept)
        # Aliceだけが見ている設定だが、正解確認用に生成
        target_image = generate_random_visual_concept(1, config['vision_dim'], device)
        
        # B. 対話実行
        result = dialogue.conduct_turn(target_image)
        
        sim = result['similarity']
        msg = result['message']
        history_similarity.append(sim)
        
        # ログ出力
        # msgはID列なので、それっぽく表示
        msg_str = " ".join([str(t) for t in msg[:3]]) + "..." 
        logger.info(f"Round {round_idx+1}: Msg='{msg_str}' -> Understanding={sim:.4f}")
        
        # 役割交代 (任意): 次はBobが話す
        # dialogue.speaker, dialogue.listener = dialogue.listener, dialogue.speaker

    # --- 5. Analysis ---
    avg_sim = np.mean(history_similarity)
    logger.info(f"\n📊 Experiment Result: Average Understanding = {avg_sim:.4f}")
    
    if avg_sim > 0.5:
        logger.info("✅ Communication Emerging: Agents are starting to share concepts.")
    else:
        logger.info("⚠️ Low Understanding: Language grounding is still difficult.")

    # 簡易グラフ (ASCII)
    logger.info("\n[Understanding Progress]")
    for sim in history_similarity:
        bar = "#" * int(sim * 20)
        logger.info(f"{sim:.2f} | {bar}")

if __name__ == "__main__":
    main()