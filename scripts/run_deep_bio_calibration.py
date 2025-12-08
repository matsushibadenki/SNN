# ファイルパス: scripts/run_deep_bio_calibration.py
# Title: Deep Bio-Calibration 実行スクリプト (データ生成修正版)
# Description:
#   指定されたSNNモデルに対して、Deep Bio-Calibration (HSEO最適化) を適用する。
#   修正: モデルアーキテクチャに応じて、テキスト用または画像用のダミーデータを
#   適切に生成するように修正し、ValueError (too many values to unpack) を解消。

import argparse
import sys
import os
import torch
import logging
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any

# プロジェクトルート設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from snn_research.conversion.bio_calibrator import DeepBioCalibrator
from snn_research.core.snn_core import SNNCore
from app.utils import get_auto_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dummy_calibration_loader(batch_size: int, samples: int, config: Dict[str, Any]) -> DataLoader:
    """
    キャリブレーション用のダミーデータローダー。
    モデル設定に基づいて入力データの形状（画像 vs テキスト）を切り替える。
    """
    arch_type = config.get("architecture_type", "unknown")
    logger.info(f"Generating calibration data for architecture: {arch_type}")

    # 画像モデルのリスト
    vision_architectures = ["spiking_cnn", "visual_cortex", "feel_snn", "sew_resnet", "hybrid_cnn_snn"]

    if arch_type in vision_architectures:
        # 画像データ (Batch, C, H, W)
        input_shape = (3, 32, 32)
        # 画像入力は通常 float
        inputs = torch.randn(samples, *input_shape)
        logger.info(f"   -> Image data generated: shape={inputs.shape}")
    else:
        # テキストモデル (Batch, SeqLen)
        # Predictive Coding, Transformer, RWKV, SFormer, SEMM など
        seq_len = 16 # ダミーのシーケンス長
        vocab_size = 100 # ダミーの語彙サイズ
        # テキスト入力は long (ID)
        inputs = torch.randint(0, vocab_size, (samples, seq_len)).long()
        logger.info(f"   -> Text data generated: shape={inputs.shape}")
        
    targets = torch.randint(0, 10, (samples,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size)

def main():
    parser = argparse.ArgumentParser(description="Run Deep Bio-Calibration for SNN models")
    parser.add_argument("--model_config", type=str, default="configs/models/micro.yaml", help="Path to model config")
    parser.add_argument("--model_path", type=str, required=False, help="Path to trained .pth model (optional)")
    parser.add_argument("--output_path", type=str, default="runs/calibrated_model.pth", help="Output path for calibrated model")
    parser.add_argument("--iterations", type=int, default=10, help="HSEO iterations")
    parser.add_argument("--particles", type=int, default=5, help="HSEO particles")
    args = parser.parse_args()

    device = get_auto_device()
    logger.info(f"Using device: {device}")

    # 1. モデルのロード
    logger.info("Loading model...")
    if not os.path.exists(args.model_config):
        logger.error(f"Config not found: {args.model_config}")
        sys.exit(1)
        
    cfg = OmegaConf.load(args.model_config)
    # 辞書変換
    model_conf = OmegaConf.to_container(cfg.model if 'model' in cfg else cfg, resolve=True)
    if not isinstance(model_conf, dict):
        logger.error("Invalid model config format.")
        sys.exit(1)
    
    # SNNCoreでラップされたモデルを作成
    snn_core = SNNCore(config=model_conf, vocab_size=100) # type: ignore
    model = snn_core.model.to(device)
    
    # 重みのロード（あれば）
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading weights from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        # キーの調整（必要なら）
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        logger.info("No weights provided or found. Using initialized weights.")

    # 2. データの準備
    logger.info("Preparing calibration data...")
    # configを渡して適切なデータを生成
    loader = create_dummy_calibration_loader(batch_size=4, samples=32, config=model_conf)

    # 3. キャリブレーション実行
    calibrator = DeepBioCalibrator(
        model=model,
        calibration_loader=loader,
        device=device,
        hseo_particles=args.particles,
        hseo_iterations=args.iterations
    )
    
    result = calibrator.calibrate()
    
    # 4. 結果の保存
    logger.info(f"Saving calibrated model to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    save_data = {
        'model_state_dict': model.state_dict(),
        'config': model_conf,
        'calibration_result': result
    }
    torch.save(save_data, args.output_path)
    logger.info("✅ Done.")

if __name__ == "__main__":
    main()
