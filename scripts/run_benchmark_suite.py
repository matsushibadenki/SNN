# ファイルパス: scripts/run_benchmark_suite.py
# (修正: learning_rate の位置を config.training.gradient_based 内に移動)

import argparse
import logging
import os
import sys
import yaml
import json
import torch
from omegaconf import OmegaConf
from typing import Any, Dict

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

from snn_research.benchmark.tasks import TaskRegistry, BenchmarkTask
from snn_research.benchmark.metrics import MetricRegistry

def ensure_benchmark_data(data_path: str) -> None:
    """
    ベンチマーク用のデータファイルが存在することを確認し、なければ作成する。
    """
    if os.path.exists(data_path):
        return

    logger.info(f"Generating dummy benchmark data at: {data_path}")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    try:
        with open(data_path, 'w', encoding='utf-8') as f:
            for i in range(100):
                sample = {
                    "text": f"This is a benchmark sample sentence number {i}. SNNs are efficient.",
                    "label": i % 2 
                }
                f.write(json.dumps(sample) + "\n")
        logger.info("Dummy benchmark data generated successfully.")
    except Exception as e:
        logger.error(f"Failed to generate dummy data: {e}")

def run_experiment(args: argparse.Namespace) -> None:
    """
    1つの実験設定を実行する
    """
    logger.info(f"Starting experiment: {args.experiment} with tag: {args.tag}")

    # 1. Configのロード
    if args.config:
        base_config = OmegaConf.load(args.config)
    else:
        # デフォルト設定（configが無い場合）
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
                    # --- ▼ 修正: learning_rate をここに移動 ▼ ---
                    "learning_rate": 1e-3,
                    # --- ▲ 修正 ▲ ---
                    "use_scheduler": False,
                    "grad_clip_norm": 1.0,
                    "use_amp": False,
                    "warmup_epochs": 0, # コンテナで参照されるため追加推奨
                    "loss": {"ce_weight": 1.0}
                }
            },
            "model": OmegaConf.load(args.model_config) if args.model_config else {},
            "data": {
                "path": "data/benchmark_data.jsonl",
                "tokenizer_name": "gpt2",
                "format": "simple_text"
            }
        })

    # CLI引数での上書き
    if args.epochs:
        base_config.training.epochs = args.epochs
    if args.batch_size:
        base_config.training.batch_size = args.batch_size
    
    # 2. 学習の実行 (train.py 連携)
    if not args.eval_only:
        # データの存在確認と生成
        data_path = OmegaConf.select(base_config, "data.path")
        if data_path:
            ensure_benchmark_data(str(data_path))

        logger.info("Running training via train.py...")
        
        if 'train' in sys.modules and hasattr(sys.modules['train'], 'train'):
            train_module = sys.modules['train']
            
            # train.train(args, config, tokenizer) を呼び出すための準備
            
            # 2a. args (Namespace) のモック作成
            class MockArgs:
                distributed = False
                data_path = None # configから読み込まれる
                task_name = args.experiment
                resume_path = None
                load_ewc_data = None
                use_astrocyte = False
                backend = "spikingjelly"
                
            train_args = MockArgs()
            
            # 2b. Tokenizer の取得 (train.container から)
            try:
                tokenizer = train_module.container.tokenizer()
            except Exception as e:
                logger.warning(f"Could not get tokenizer from container: {e}. Using AutoTokenizer fallback.")
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")

            # 2c. train 関数実行
            try:
                # DictConfig 型であることを保証
                if not isinstance(base_config, (dict, OmegaConf.get_type("DictConfig"))): # type: ignore
                     base_config = OmegaConf.create(base_config)
                
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

    # 3. 評価 (Evaluation) - 簡易実装
    if args.eval_only:
         logger.info("Performing evaluation-only steps...")
         print(f"Evaluation for {args.experiment} completed (Simulated).")

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
