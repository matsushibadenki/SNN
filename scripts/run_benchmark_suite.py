# ファイルパス: scripts/run_benchmark_suite.py
# (修正: モデル設定の階層構造を正規化し、SNNCoreに正しい辞書が渡るように修正)

import argparse
import logging
import os
import sys
import yaml
import json
import torch
import random
from omegaconf import OmegaConf
from typing import Any, Dict
from PIL import Image
import numpy as np

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# scripts/runners ディレクトリをパスに追加して、trainモジュールをインポート可能にする
runners_dir = os.path.join(project_root, 'scripts', 'runners')
if runners_dir not in sys.path:
    sys.path.append(runners_dir)

try:
    import train # type: ignore[import-not-found]
except ImportError as e:
    logger.error(f"trainモジュールのインポートに失敗しました: {e}")
    pass

def ensure_text_benchmark_data(data_path: str) -> None:
    """ベンチマーク用のダミーテキストデータを作成"""
    if os.path.exists(data_path): return
    logger.info(f"Generating dummy TEXT benchmark data at: {data_path}")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    try:
        with open(data_path, 'w', encoding='utf-8') as f:
            for i in range(100):
                sample = {
                    "text": f"This is a benchmark sample sentence number {i}. SNNs are efficient.",
                    "label": i % 2 
                }
                f.write(json.dumps(sample) + "\n")
        logger.info("Dummy text data generated.")
    except Exception as e:
        logger.error(f"Failed to generate text data: {e}")

def ensure_image_benchmark_data(data_path: str) -> None:
    """ベンチマーク用のダミー画像データを作成"""
    if os.path.exists(data_path): return
    
    data_dir = os.path.dirname(data_path)
    img_dir = os.path.join(data_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    logger.info(f"Generating dummy IMAGE benchmark data at: {data_path}")
    
    try:
        with open(data_path, 'w', encoding='utf-8') as f:
            for i in range(20): # 20枚生成
                img_name = f"dummy_{i}.jpg"
                img_path = os.path.join(img_dir, img_name)
                
                arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(img_path)
                
                sample = {
                    "text": f"Dummy caption for image {i}",
                    "image": os.path.join("images", img_name),
                    "label": i % 10
                }
                f.write(json.dumps(sample) + "\n")
        logger.info("Dummy image data generated.")
    except Exception as e:
        logger.error(f"Failed to generate image data: {e}")

def run_experiment(args: argparse.Namespace) -> None:
    logger.info(f"Starting experiment: {args.experiment} with tag: {args.tag}")

    # 1. モデル設定のロードとアーキテクチャ判定
    model_conf_dict = {}
    if args.model_config and os.path.exists(args.model_config):
        model_conf_loaded = OmegaConf.load(args.model_config)
        
        # --- ▼ 修正: 階層構造の正規化 ▼ ---
        if 'model' in model_conf_loaded:
            # {model: {architecture_type: ...}} の場合、中身を取り出す
            model_conf_dict = OmegaConf.to_container(model_conf_loaded.model, resolve=True) # type: ignore
        else:
            # フラットな構造の場合
            model_conf_dict = OmegaConf.to_container(model_conf_loaded, resolve=True) # type: ignore
        # --- ▲ 修正 ▲ ---
    
    arch_type = model_conf_dict.get("architecture_type", "unknown")
    logger.info(f"Detected architecture type: {arch_type}")

    is_vision = arch_type in ["spiking_cnn", "visual_cortex", "feel_snn", "sew_resnet", "hybrid_cnn_snn"]
    data_format = "image_text" if is_vision else "simple_text"
    
    logger.info(f"Selected data format: {data_format}")

    # 2. Configの構築
    if args.config:
        base_config = OmegaConf.load(args.config)
    else:
        base_config = OmegaConf.create({
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "optimizer": "adam",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "log_dir": "runs/benchmark",
                "paradigm": "gradient_based",
                "gradient_based": {
                    "type": "standard",
                    "learning_rate": 1e-3,
                    "use_scheduler": False,
                    "grad_clip_norm": 1.0,
                    "use_amp": False,
                    "warmup_epochs": 0,
                    "loss": {"ce_weight": 1.0}
                }
            },
            "model": model_conf_dict, # 修正済みの辞書を渡す
            "data": {
                "path": "data/benchmark_data.jsonl",
                "tokenizer_name": "gpt2",
                "format": data_format
            }
        })

    # CLI引数での上書き
    if args.epochs: base_config.training.epochs = args.epochs
    if args.batch_size: base_config.training.batch_size = args.batch_size
    
    # 3. データの準備
    data_path = str(base_config.data.path)
    if not os.path.exists(data_path):
        if is_vision:
            ensure_image_benchmark_data(data_path)
        else:
            ensure_text_benchmark_data(data_path)

    # 4. 学習の実行
    if not args.eval_only:
        logger.info("Running training via train.py...")
        
        if 'train' in sys.modules and hasattr(sys.modules['train'], 'train'):
            train_module = sys.modules['train']
            
            class MockArgs:
                distributed = False
                data_path = None
                task_name = args.experiment
                resume_path = None
                load_ewc_data = None
                use_astrocyte = False
                backend = "spikingjelly"
                
            train_args = MockArgs()
            
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except Exception:
                tokenizer = None

            try:
                if not isinstance(base_config, (dict, OmegaConf.get_type("DictConfig"))): # type: ignore
                     base_config = OmegaConf.create(base_config)
                
                # train関数呼び出し
                train_module.train(train_args, base_config, tokenizer) # type: ignore
                logger.info("Training completed via train.py.")
            except Exception as e:
                logger.error(f"Training failed: {e}", exc_info=True)
                if not args.force_continue:
                     raise e
        else:
            logger.warning("train.train function not found. Skipping training execution.")
    else:
        logger.info("Skipping training (eval_only=True).")

def main() -> None:
    parser = argparse.ArgumentParser(description='SNN Benchmark Suite')
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--config', type=str, help='Path to base training config')
    parser.add_argument('--model_config', type=str, help='Path to model architecture config')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--tag', type=str, default='default', help='Tag for the run')
    parser.add_argument('--eval_only', action='store_true', help='Skip training')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint for eval')
    parser.add_argument('--model_type', type=str, default='SNN', help='SNN or ANN')
    parser.add_argument('--force_continue', action='store_true', help='Continue even if training fails')
    
    args = parser.parse_args()
    
    run_experiment(args)

if __name__ == "__main__":
    main()
