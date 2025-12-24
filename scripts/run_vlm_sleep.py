# ファイルパス: scripts/run_vlm_sleep.py
# (Phase 4: Sleep Consolidation Demo - Fixed)
# Title: VLM Sleep & Dreaming Demo
# Description: ログ出力を標準出力に強制し、実行状況を可視化する。

import sys
import os
import torch
import logging
import numpy as np

# プロジェクトルートをPythonパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.models.transformer.spiking_vlm import SpikingVLM
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator

# ロガー設定: 標準出力に強制的に出す
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_trained_model(checkpoint_path, device):
    print(f"📂 Loading VLM from: {checkpoint_path}") # 強制出力
    logger.info(f"📂 Loading VLM from: {checkpoint_path}")
    
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
    print("🚀 Starting Sleep Simulation...") # 強制出力
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/vlm/spiking_vlm_epoch_2.pt"
    
    # 1. モデルロード
    brain = load_trained_model(checkpoint_path, device)
    
    # 2. 睡眠コンソリデータ初期化
    sleeper = SleepConsolidator(memory_system={}, target_brain_model=brain, dream_rate=0.05)
    
    # 3. 日中の活動シミュレーション
    logger.info("☀️ DAYTIME: Accumulating experiences...")
    for i in range(5):
        sleeper.experience_buffer.append({"type": "visual_event", "content_id": i})
    
    # 4. 夜間：睡眠サイクル
    logger.info("🌙 NIGHTTIME: Entering REM sleep (Dreaming)...")
    
    # 50サイクルの夢を見る
    results = sleeper.perform_sleep_cycle(duration_cycles=50)
    
    # 5. 結果分析
    clarity_history = results["loss_history"]
    avg_clarity = np.mean(clarity_history) if clarity_history else 0.0
    
    logger.info(f"💤 Sleep Cycle Completed.")
    logger.info(f"   - Dreams Replayed: {results['dreams_replayed']}")
    logger.info(f"   - Average Dream Clarity: {avg_clarity:.4f}")
    
    if avg_clarity > 0.0:
        print("✅ SUCCESS: The brain generated internal dreams.")
    else:
        print("⚠️ The brain activity was silent.")

# ここが欠落していると実行されません
if __name__ == "__main__":
    main()
