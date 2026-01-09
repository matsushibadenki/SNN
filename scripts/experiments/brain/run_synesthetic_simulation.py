# ファイルパス: scripts/experiments/brain/run_synesthetic_simulation.py
# 日本語タイトル: Run Synesthetic Brain Simulation
# 目的: Brain v4 (五感統合モデル) の動作検証。
#       視覚、聴覚、触覚、嗅覚のダミーデータを生成し、統合処理と学習ステップを実行する。

from snn_research.utils.efficiency_profiler import EfficiencyProfiler
from snn_research.models.experimental.brain_v4 import SynestheticBrain
import os
import sys
import torch
import torch.nn as nn
import logging
from typing import Dict

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())


# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SynesthesiaSim")


def generate_dummy_sensory_data(batch_size: int, seq_len: int, config: Dict) -> Dict[str, torch.Tensor]:
    """各モダリティのダミーデータを生成"""
    data = {}
    device = config['device']

    # Vision: (B, T, C, H, W) or (B, C, H, W) based on encoder
    # ここでは簡易化のため特徴量ベース (B, seq_len, input_dim) とする
    if 'vision' in config['sensory']:
        dim = config['sensory']['vision']
        data['vision'] = torch.randn(batch_size, seq_len, dim, device=device)

    # Audio
    if 'audio' in config['sensory']:
        dim = config['sensory']['audio']
        data['audio'] = torch.randn(batch_size, seq_len, dim, device=device)

    # Tactile (触覚)
    if 'tactile' in config['sensory']:
        dim = config['sensory']['tactile']
        # 触覚はスパースであることが多いが、ここではランダム
        data['tactile'] = torch.randn(batch_size, seq_len, dim, device=device)

    # Olfactory (嗅覚)
    if 'olfactory' in config['sensory']:
        dim = config['sensory']['olfactory']
        data['olfactory'] = torch.abs(torch.randn(
            batch_size, seq_len, dim, device=device))  # 濃度なので正の値

    return data


def main():
    logger.info("🚀 Starting Synesthetic Brain Simulation...")

    # --- Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        'device': device,
        'vocab_size': 1000,
        'd_model': 128,
        'time_steps': 8,
        'sensory': {
            'vision': 784,
            'audio': 64,
            'tactile': 32,   # 新規追加
            'olfactory': 16  # 新規追加
        }
    }

    # --- Model Initialization ---
    logger.info(f"🧠 Initializing Brain v4 on {device}...")
    model = SynestheticBrain(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_layers=2,  # デモ用に軽量化
        time_steps=config['time_steps'],
        tactile_dim=config['sensory']['tactile'],
        olfactory_dim=config['sensory']['olfactory'],
        device=device
    )

    # 最適化設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    profiler = EfficiencyProfiler()

    # --- Simulation Loop ---
    num_steps = 5
    batch_size = 4
    seq_len = 1  # Brain v4のEncoderは入力を即時処理する想定(内部でTimeSteps展開)

    model.train()

    for step in range(num_steps):
        logger.info(f"⚡ Step {step+1}/{num_steps}")

        # 1. Generate Inputs (五感入力 + テキスト思考コンテキスト)
        sensory_data = generate_dummy_sensory_data(batch_size, seq_len, config)

        # テキスト入力（思考の種）
        text_input = torch.randint(
            0, config['vocab_size'], (batch_size, 16)).to(device)

        # 正解ラベル（次の単語予測タスクと仮定）
        targets = torch.randint(0, config['vocab_size'], (batch_size, 16)).to(
            device)  # 長さはtext_input + sensory_context分ずれるが簡易化

        # 2. Forward Pass
        profiler.start_measurement()

        # Brain v4 forward: 五感を全て渡す
        logits = model(
            text_input=text_input,
            image_input=sensory_data['vision'],
            audio_input=sensory_data['audio'],
            tactile_input=sensory_data['tactile'],
            olfactory_input=sensory_data['olfactory']
        )

        # Logits shape: [B, Total_Seq_Len, Vocab]
        # ターゲットとサイズを合わせるための簡易スライス（実際はシフトが必要）
        output_len = logits.size(1)
        target_len = targets.size(1)
        min_len = min(output_len, target_len)

        loss = criterion(logits[:, :min_len, :].reshape(-1, config['vocab_size']),
                         targets[:, :min_len].reshape(-1))

        # 3. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        profiler.end_measurement()

        logger.info(f"   Loss: {loss.item():.4f}")
        logger.info(f"   Context Length: {output_len} (Sensory + Text)")

    # --- Generative Demo ---
    logger.info("🎨 Testing Generative Capability (Cross-Modal)...")
    model.eval()

    # 「画像」を見て「言葉」を発するデモ
    test_image = torch.randn(1, 1, config['sensory']['vision'], device=device)
    start_token = 101  # BOS

    generated_ids = model.generate(
        image_input=test_image,
        start_token_id=start_token,
        max_new_tokens=10
    )

    logger.info(f"   Generated Tokens from Visual Input: {generated_ids}")
    logger.info("✅ Simulation completed successfully.")


if __name__ == "__main__":
    main()
