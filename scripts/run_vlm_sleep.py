# ファイルパス: scripts/run_vlm_sleep.py
# (Phase 4: Sleep Consolidation Demo)
# Title: VLM Sleep & Dreaming Demo
# Description:
#   学習済みSpikingVLMに対し、睡眠サイクル(SleepConsolidator)を適用する。
#   視覚野にノイズを与え、言語野がそこから意味を見出す「夢」のプロセスを観察する。

import sys
import os
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np

# プロジェクトルートをPythonパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.models.transformer.spiking_vlm import SpikingVLM
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator

# ロガー設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

def load_trained_model(checkpoint_path, device):
    """モデル構築"""
    logger.info(f"📂 Loading VLM from: {checkpoint_path}")
    
    # Configは学習時と合わせる
    vision_config = {"architecture_type": "spiking_cnn", "input_channels": 3, "features": 128, "time_steps": 4, "layers": [64, 128, 128]}
    language_config = {"architecture_type": "spiking_transformer", "vocab_size": 30522, "d_model": 256, "num_layers": 4, "num_heads": 4, "time_steps": 4, "max_len": 64}
    projector_config = {"visual_dim": 128, "use_bitnet": True}
    
    model = SpikingVLM(
        vocab_size=30522,
        vision_config=vision_config,
        language_config=language_config,
        projector_config=projector_config
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info("✅ Weights loaded.")
    else:
        logger.warning("⚠️ Checkpoint not found. Initializing with random weights.")
        
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/vlm/spiking_vlm_epoch_2.pt"
    
    # 1. モデルロード
    brain = load_trained_model(checkpoint_path, device)
    
    # 2. 睡眠コンソリデータ（海馬モデル）の初期化
    # memory_systemは今回はダミー
    sleeper = SleepConsolidator(memory_system={}, target_brain_model=brain, dream_rate=0.05)
    
    # 3. 日中の活動（エピソード記憶の蓄積シミュレーション）
    logger.info("☀️ DAYTIME: Accumulating experiences...")
    for i in range(5):
        sleeper.experience_buffer.append({"type": "visual_event", "content_id": i})
    
    # 4. 夜間：睡眠サイクルの実行
    logger.info("🌙 NIGHTTIME: Entering REM sleep (Dreaming)...")
    
    # 50サイクルの夢を見る
    results = sleeper.perform_sleep_cycle(duration_cycles=50)
    
    # 5. 結果の分析
    clarity_history = results["loss_history"]
    avg_clarity = np.mean(clarity_history)
    
    logger.info(f"💤 Sleep Cycle Completed.")
    logger.info(f"   - Dreams Replayed: {results['dreams_replayed']}")
    logger.info(f"   - Average Dream Clarity: {avg_clarity:.4f}")
    
    if avg_clarity > 0.0:
        logger.info("✅ SUCCESS: The brain generated internal activity (dreams) without external input.")
        logger.info("   (Higher clarity means the network settled into stable attractor states corresponding to learned concepts.)")
    else:
        logger.warning("⚠️ The brain activity was silent. Check noise injection.")

if __name__ == "__main__":
    main()