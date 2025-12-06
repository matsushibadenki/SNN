# ファイルパス: scripts/run_benchmark_suite.py
# (修正: logging.basicConfig に stream=sys.stdout を追加)

import argparse
import logging
import os
import sys
import yaml
import json
import torch
import random
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Any, Dict, cast
from PIL import Image
import numpy as np

# --- 修正: ストリームを標準出力に設定 ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

runners_dir = os.path.join(project_root, 'scripts', 'runners')
if runners_dir not in sys.path:
    sys.path.append(runners_dir)

try:
    import train # type: ignore [import-not-found]
except ImportError as e:
    logger.error(f"trainモジュールのインポートに失敗しました: {e}")
    pass

def ensure_text_benchmark_data(data_path: str) -> None:
    if os.path.exists(data_path): return
    logger.info(f"Generating dummy TEXT benchmark data at: {data_path}")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    try:
        with open(data_path, 'w', encoding='utf-8') as f:
            for i in range(100):
                sample = {"text": f"Benchmark sample {i}. SNNs are efficient.", "label": i % 2}
                f.write(json.dumps(sample) + "\n")
    except Exception as e:
        logger.error(f"Failed to generate text data: {e}")

def ensure_image_benchmark_data(data_path: str) -> None:
    if os.path.exists(data_path): return
    data_dir = os.path.dirname(data_path)
    img_dir = os.path.join(data_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    logger.info(f"Generating dummy IMAGE benchmark data at: {data_path}")
    try:
        with open(data_path, 'w', encoding='utf-8') as f:
            for i in range(20):
                img_name = f"dummy_{i}.jpg"
                img_path = os.path.join(img_dir, img_name)
                arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(img_path)
                sample = {"text": f"Dummy caption {i}", "image": os.path.join("images", img_name), "label": i % 10}
                f.write(json.dumps(sample) + "\n")
    except Exception as e:
        logger.error(f"Failed to generate image data: {e}")

def run_experiment(args: argparse.Namespace) -> None:
    logger.info(f"Starting experiment: {args.experiment} with tag: {args.tag}")

    if not args.model_config:
        if "cifar10" in args.experiment:
            args.model_config = "configs/experiments/cifar10_spikingcnn_config.yaml"
        else:
            args.model_config = "configs/models/micro.yaml"
        logger.info(f"No model config provided. Using default: {args.model_config}")

    model_conf_dict: Dict[str, Any] = {}
    if args.model_config and os.path.exists(args.model_config):
        model_conf_loaded = OmegaConf.load(args.model_config)
        
        if 'architecture_type' in model_conf_loaded:
            model_conf_dict = cast(Dict[str, Any], OmegaConf.to_container(model_conf_loaded, resolve=True))
        elif 'model' in model_conf_loaded:
            model_conf_dict = cast(Dict[str, Any], OmegaConf.to_container(model_conf_loaded.model, resolve=True)) # type: ignore
        else:
            logger.warning(f"Could not find 'architecture_type' or 'model' key in config. Using root.")
            model_conf_dict = cast(Dict[str, Any], OmegaConf.to_container(model_conf_loaded, resolve=True))
    
    if not model_conf_dict:
        logger.error(f"Failed to load model config from {args.model_config}. Dictionary is empty.")
        return
        
    arch_type = model_conf_dict.get("architecture_type", "unknown")
    logger.info(f"Detected architecture type: {arch_type}")

    is_vision = arch_type in ["spiking_cnn", "visual_cortex", "feel_snn", "sew_resnet", "hybrid_cnn_snn"]
    data_format = "image_text" if is_vision else "simple_text"
    logger.info(f"Selected data format: {data_format}")

    # Config構築
    base_config = OmegaConf.create({
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "adam",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "log_dir": "runs/benchmark",
            "eval_interval": 1,
            "log_interval": 1,
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
        "model": model_conf_dict,
        "data": {
            "path": "data/benchmark_data.jsonl",
            "tokenizer_name": "gpt2",
            "format": data_format
        }
    })

    if args.epochs: base_config.training.epochs = args.epochs
    if args.batch_size: base_config.training.batch_size = args.batch_size
    if args.config: 
        ext_conf = OmegaConf.load(args.config)
        base_config = cast(DictConfig, OmegaConf.merge(base_config, ext_conf))

    data_path = str(base_config.data.path)
    if not os.path.exists(data_path):
        if is_vision: ensure_image_benchmark_data(data_path)
        else: ensure_text_benchmark_data(data_path)

    if not args.eval_only:
        logger.info("Running training via train.py...")
        if 'train' in sys.modules:
            train_module = sys.modules['train']
            
            # --- 修正: argparse.Namespace を使用して引数を模倣 ---
            mock_args = argparse.Namespace(
                distributed=False,
                data_path=None,
                task_name=args.experiment,
                resume_path=None,
                load_ewc_data=None,
                use_astrocyte=False,
                backend="spikingjelly"
            )
            # ---------------------------------------------------

            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except: tokenizer = None

            try:
                if not isinstance(base_config, (dict, DictConfig)):
                     base_config = OmegaConf.create(base_config)
                
                train_module.train(mock_args, base_config, tokenizer) # type: ignore
                logger.info("Training completed.")
            except Exception as e:
                logger.error(f"Training failed: {e}", exc_info=True)
                if not args.force_continue: raise e
        else:
            logger.warning("train module not found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--config')
    parser.add_argument('--model_config')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tag', default='default')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--model_path')
    parser.add_argument('--model_type', default='SNN')
    parser.add_argument('--force_continue', action='store_true')
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main()
