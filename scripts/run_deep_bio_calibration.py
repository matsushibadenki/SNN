# ファイルパス: scripts/run_deep_bio_calibration.py
# Title: Deep Bio-Calibration 実行スクリプト
# Description:
#   指定されたSNNモデルに対して、Deep Bio-Calibration (HSEO最適化) を適用する。
#   既存のモデルをロードし、キャリブレーションデータセットを用いて閾値を微調整し、
#   最適化されたモデルを保存する。

import argparse
import sys
import os
import torch
import logging
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

# プロジェクトルート設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from snn_research.conversion.bio_calibrator import DeepBioCalibrator
from snn_research.core.snn_core import SNNCore
from app.utils import get_auto_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dummy_calibration_loader(batch_size: int = 4, samples: int = 32, input_shape=(3, 32, 32)):
    """キャリブレーション用のダミーデータローダー"""
    # 画像タスクを想定
    inputs = torch.randn(samples, *input_shape)
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
    
    # SNNCoreでラップされたモデルを作成
    # 実運用では実際の vocab_size や num_classes を設定する
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
    # 実運用では DataContainer などから実際の検証データセットの一部を取得する
    logger.info("Preparing calibration data...")
    # 簡易的に画像入力(3, 32, 32)を想定
    loader = create_dummy_calibration_loader(input_shape=(3, 32, 32))

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