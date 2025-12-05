# ファイルパス: scripts/report_sparsity_and_T.py
# Title: スパイク効率性レポート生成 (修正版)
# Description: 指定されたモデル設定に基づいてSNNを構築し、ダミーデータを用いてスパイク率やタイムステップ数を計測する。

import sys
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import argparse
import logging
import json

# プロジェクトルートの解決
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from snn_research.core.snn_core import SNNCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def measure_efficiency(model_config_path: str, data_path: str):
    logger.info(f"Loading config from {model_config_path}")
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Config not found: {model_config_path}")
        
    cfg = OmegaConf.load(model_config_path)
    
    # --- ▼ 修正: 階層構造の正規化 ▼ ---
    if 'model' in cfg:
        model_params = OmegaConf.to_container(cfg.model, resolve=True)
    else:
        model_params = OmegaConf.to_container(cfg, resolve=True)
    # --- ▲ 修正 ▲ ---

    logger.info("Building SNN model...")
    # SNNCoreには辞書形式で渡す
    model = SNNCore(model_params) # type: ignore
    model.eval()
    
    # ダミー入力の作成 (モデルタイプに合わせて)
    arch_type = model_params.get("architecture_type", "unknown") # type: ignore
    logger.info(f"Architecture: {arch_type}")
    
    # ... (以降のロジックは変更なし) ...
    # 簡易的にテキスト/画像入力を切り分け
    if arch_type in ["spiking_cnn", "visual_cortex", "feel_snn"]:
        # 画像入力 (B, C, H, W)
        dummy_input = torch.randn(1, 3, 32, 32)
    else:
        # テキスト入力 (B, L)
        dummy_input = torch.randint(0, 100, (1, 128))

    logger.info(f"Running inference with input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    total_spikes = model.get_total_spikes()
    logger.info(f"Total Spikes: {total_spikes}")
    
    # 詳細なレポート出力などのロジック...
    print(f"Efficiency Report:\n  Architecture: {arch_type}\n  Total Spikes: {total_spikes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/smoke_test_data.jsonl")
    args = parser.parse_args()
    
    measure_efficiency(args.model_config, args.data_path)
