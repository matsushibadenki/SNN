# ファイルパス: scripts/training/train.py
# タイトル: SNN学習実行スクリプト (修正版: Import Error解消)
# 修正内容:
# 1. 存在しない 'scripts.data.data_preparation' のインポートを削除。
# 2. '--model_config ...' のフォールバックロジックなどの堅牢性は維持。

from snn_research.training.bio_trainer import BioRLTrainer
from snn_research.training.trainers import (
    BreakthroughTrainer,
    ParticleFilterTrainer,
    DistillationTrainer,
    SelfSupervisedTrainer,
    PhysicsInformedTrainer,
    ProbabilisticTrainer
)
from snn_research.core.snn_core import SNNCore
from snn_research.data.datasets import get_dataset_class, DistillationDataset, DataFormat, SNNBaseDataset
from app.utils import get_auto_device, collate_fn
from app.containers import TrainingContainer
import sys
import os
import argparse
import logging
import traceback
from typing import Optional, Dict, Any, cast, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from omegaconf import DictConfig, OmegaConf, ListConfig

# プロジェクトルートの設定
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ロガー設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(args: argparse.Namespace, config: DictConfig, tokenizer: Optional[PreTrainedTokenizerBase]) -> None:
    """
    学習プロセスのメインルーチン。
    DIコンテナを用いてモデル、オプティマイザ、トレーナーを構築し、学習ループを実行する。
    """
    # --- 1. DIコンテナのセットアップ ---
    container = TrainingContainer()

    # モデル設定用変数の初期化
    model_conf_dict: Dict[str, Any] = {}
    full_conf_dict: Dict[str, Any] = {}

    # コンフィグの辞書変換とDIコンテナへの注入
    if config:
        # OmegaConf -> primitive dict (resolve=Trueで補間解決)
        full_conf_dict = cast(
            Dict[str, Any], OmegaConf.to_container(config, resolve=True))

        if isinstance(full_conf_dict, dict):
            # 全体の設定を注入
            container.config.from_dict(full_conf_dict)

            # モデル設定の抽出ロジック
            # 優先順位: config['model'] > config (ルート)
            temp_model = full_conf_dict.get('model', {})

            # modelキーの中に architecture_type があればそれを採用
            if isinstance(temp_model, dict) and 'architecture_type' in temp_model:
                model_conf_dict = temp_model
            # なければルートレベルを探す
            elif 'architecture_type' in full_conf_dict:
                model_conf_dict = full_conf_dict
            else:
                logger.warning(
                    "⚠️ 'architecture_type' not explicitly found in 'model' dict. Using root config or empty dict.")
                model_conf_dict = temp_model if isinstance(
                    temp_model, dict) else full_conf_dict

            arch = model_conf_dict.get('architecture_type', 'unknown')
            logger.info(f"🔧 Train Config Prepared. Architecture: {arch}")

        else:
            logger.error(
                f"❌ Config is not a dictionary, got {type(full_conf_dict)}")
            return

    # --- 2. デバイスと分散学習設定 ---
    is_distributed = args.distributed
    rank = int(os.environ.get("LOCAL_RANK", -1))
    device = f'cuda:{rank}' if is_distributed and torch.cuda.is_available(
    ) else get_auto_device()
    logger.info(
        f"🚀 Using device: {device} (Distributed: {is_distributed}, Rank: {rank})")

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)

    # --- 3. 学習パラダイムの決定 ---
    paradigm = config.training.paradigm
    logger.info(f"📚 Learning Paradigm: {paradigm}")

    # トレーナー型ヒント
    trainer: Union[
        BreakthroughTrainer,
        BioRLTrainer,
        ParticleFilterTrainer,
        DistillationTrainer,
        SelfSupervisedTrainer,
        PhysicsInformedTrainer,
        ProbabilisticTrainer
    ]

    # --- A. 生物学的学習パラダイム ---
    if paradigm.startswith("bio-"):
        if paradigm == "bio-causal-sparse":
            container.config.training.biologically_plausible.adaptive_causal_sparsification.enabled.from_value(
                True)
            trainer = container.bio_rl_trainer()
            cast(BioRLTrainer, trainer).train(
                num_episodes=config.training.epochs)

        elif paradigm == "bio-particle-filter":
            trainer = container.particle_filter_trainer()
            dummy_data = torch.rand(1, 10, device=device)
            dummy_targets = torch.rand(1, 2, device=device)
            for epoch in range(config.training.epochs):
                loss = cast(ParticleFilterTrainer, trainer).train_step(
                    dummy_data, dummy_targets)
                logger.info(
                    f"Epoch {epoch+1}/{config.training.epochs}: Particle Filter Loss = {loss:.4f}")

        elif paradigm == "bio-probabilistic-hebbian":
            prob_trainer = container.probabilistic_trainer() if hasattr(
                container, 'probabilistic_trainer') else container.bio_rl_trainer()
            cast(BioRLTrainer, prob_trainer).train(
                num_episodes=config.training.epochs)

        else:
            raise ValueError(f"Unknown bio paradigm: {paradigm}")

        logger.info("✅ Bio-inspired training completed.")
        return

    # --- B. 勾配ベース学習パラダイム ---
    elif paradigm in ["gradient_based", "self_supervised", "physics_informed", "probabilistic_ensemble"]:

        # 1. データセット準備
        grad_config = config.training.get("gradient_based", {})
        grad_type = grad_config.get("type", "standard")
        is_distillation = (
            paradigm == "gradient_based" and grad_type == "distillation")

        data_path = args.data_path or OmegaConf.select(
            config, "data.path", default="data/default_data.jsonl")

        # WikiTextの自動ダウンロード機能は scripts.data が存在しないため無効化
        if "wikitext-103" in str(data_path) and not os.path.exists(str(data_path)):
            logger.warning(
                "⚠️ WikiText-103 dataset is missing and automatic download is disabled (module missing).")
            # prepare_wikitext_data()

        if not os.path.exists(str(data_path)):
            if "smoke_test_data.jsonl" in str(data_path):
                logger.info(f"⚠️ Creating dummy data at: {data_path}")
                os.makedirs(os.path.dirname(str(data_path)), exist_ok=True)
                with open(str(data_path), 'w') as f:
                    import json
                    for i in range(10):
                        f.write(json.dumps(
                            {"text": f"This is a smoke test sample {i}."}) + "\n")
            else:
                raise FileNotFoundError(f"Data file not found: {data_path}")

        if tokenizer is None:
            logger.warning(
                "Tokenizer is None. Loading default 'gpt2' tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        DatasetClass = get_dataset_class(DataFormat(config.data.format))
        max_seq_len = OmegaConf.select(config, "model.time_steps", default=128)

        dataset: SNNBaseDataset
        if is_distillation:
            data_dir = os.path.dirname(str(data_path))
            distill_jsonl_path = os.path.join(
                data_dir, "distillation_data.jsonl")

            if os.path.exists(distill_jsonl_path):
                logger.info(
                    f"Using DistillationDataset from {distill_jsonl_path}")
                dataset = DistillationDataset(
                    file_path=distill_jsonl_path,
                    data_dir=data_dir,
                    tokenizer=tokenizer,
                    max_seq_len=int(max_seq_len)
                )
            else:
                logger.warning(
                    "⚠️ Distillation dataset not found. Falling back to standard dataset.")
                is_distillation = False
                dataset = DatasetClass(file_path=str(
                    data_path), tokenizer=tokenizer, max_seq_len=int(max_seq_len))
        else:
            dataset = DatasetClass(file_path=str(
                data_path), tokenizer=tokenizer, max_seq_len=int(max_seq_len))

        logger.info(f"Dataset loaded: {len(dataset)} samples.")

        split_ratio = float(OmegaConf.select(
            config, "data.split_ratio", default=0.1))
        train_size = int((1.0 - split_ratio) * len(dataset))
        val_size = len(dataset) - train_size
        if train_size <= 0:
            train_size = len(dataset)
            val_size = 0

        train_dataset_split, val_dataset_split = random_split(
            dataset, [train_size, val_size])

        train_sampler: Optional[DistributedSampler] = DistributedSampler(
            train_dataset_split) if is_distributed else None

        collate_fn_instance = collate_fn(tokenizer, is_distillation)
        batch_size = int(config.training.batch_size)

        train_loader = DataLoader(
            train_dataset_split,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_fn_instance
        )
        val_loader = DataLoader(
            val_dataset_split,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_instance
        )

        # 2. モデル構築
        vocab_size = int(OmegaConf.select(
            config, "data.max_vocab_size", default=1000))
        logger.info(
            f"🏗️ Initializing SNNCore with backend={args.backend}, vocab_size={vocab_size}")

        try:
            snn_model: nn.Module = SNNCore(
                config=model_conf_dict,
                vocab_size=vocab_size,
                backend=args.backend
            )
        except ValueError as e:
            logger.error(f"❌ Failed to initialize SNNCore: {e}")
            logger.error(
                f"Debug: model_conf_dict keys: {list(model_conf_dict.keys())}")
            raise e

        snn_model.to(device)

        if is_distributed:
            snn_model = DDP(snn_model, device_ids=[
                            rank], find_unused_parameters=True)

        optimizer = container.optimizer(params=snn_model.parameters())
        use_scheduler = bool(config.training.gradient_based.use_scheduler)
        scheduler = container.scheduler(
            optimizer=optimizer) if use_scheduler else None

        # 3. トレーナーの選択と初期化
        if paradigm == "gradient_based":
            if is_distillation:
                logger.info("🎓 Using DistillationTrainer")
                trainer = container.distillation_trainer(
                    model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank
                )
            else:
                logger.info("👨‍🏫 Using Standard BreakthroughTrainer")
                trainer = container.standard_trainer(
                    model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank
                )
        elif paradigm == "self_supervised":
            trainer = container.self_supervised_trainer(
                model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank
            )
        elif paradigm == "physics_informed":
            trainer = container.physics_informed_trainer(
                model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank
            )
        else:  # probabilistic_ensemble
            trainer = container.probabilistic_ensemble_trainer(
                model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank
            )

        bt_trainer = cast(BreakthroughTrainer, trainer)

        if args.load_ewc_data:
            bt_trainer.load_ewc_data(args.load_ewc_data)

        start_epoch = 0
        if args.resume_path:
            start_epoch = bt_trainer.load_checkpoint(args.resume_path)
            logger.info(f"Resumed from epoch {start_epoch}")

        # 4. 学習ループ
        total_epochs = int(config.training.epochs)
        eval_interval = int(config.training.eval_interval)
        log_interval = int(config.training.log_interval)
        log_dir = str(config.training.log_dir)

        for epoch in range(start_epoch, total_epochs):
            bt_trainer.train_epoch(train_loader, epoch)

            if rank in [-1, 0] and (epoch % eval_interval == 0 or epoch == total_epochs - 1):
                val_metrics = bt_trainer.evaluate(val_loader, epoch)

                if epoch % log_interval == 0:
                    checkpoint_path = os.path.join(
                        log_dir, f"checkpoint_epoch_{epoch}.pth")

                    # 保存用にConfigをdict化
                    save_conf_dict: Dict[str, Any]
                    if isinstance(config.model, DictConfig):
                        save_conf_dict = cast(
                            Dict[str, Any], OmegaConf.to_container(config.model, resolve=True))
                    else:
                        # Fallback
                        save_conf_dict = model_conf_dict

                    bt_trainer.save_checkpoint(
                        path=checkpoint_path,
                        epoch=epoch,
                        metric_value=val_metrics.get('total', float('inf')),
                        tokenizer_name=str(config.data.tokenizer_name),
                        config=save_conf_dict
                    )

        if args.task_name and rank in [-1, 0]:
            ewc_weight = float(OmegaConf.select(
                config, "training.gradient_based.loss.ewc_weight", default=0.0))
            if ewc_weight > 0:
                logger.info(
                    f"🔒 Computing EWC Fisher Matrix for task '{args.task_name}'...")
                bt_trainer._compute_ewc_fisher_matrix(
                    train_loader, args.task_name)  # type: ignore[attr-defined]

    else:
        raise ValueError(f"Unknown training paradigm: '{paradigm}'.")

    logger.info("✅ 学習が完了しました。")


def main() -> None:
    parser = argparse.ArgumentParser(description="SNN Training Runner")
    parser.add_argument(
        "--config", type=str, default="configs/templates/base_config.yaml", help="Base config file")
    parser.add_argument("--model_config", type=str,
                        help="Model architecture config file")
    parser.add_argument("--data_path", type=str, help="Path to dataset file")
    parser.add_argument("--task_name", type=str, help="Task name for EWC")
    parser.add_argument("--override_config", type=str, action='append',
                        help="Override config (e.g. training.epochs=10)")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training (DDP)")
    parser.add_argument("--resume_path", type=str,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--load_ewc_data", type=str,
                        help="Path to EWC data file")
    parser.add_argument("--use_astrocyte", action="store_true",
                        help="Enable Astrocyte Network")
    parser.add_argument("--paradigm", type=str,
                        help="Override training paradigm")
    parser.add_argument("--backend", type=str,
                        default="spikingjelly", help="SNN backend")
    args = parser.parse_args()

    try:
        # 1. Base Configの読み込み
        base_path = "configs/templates/base_config.yaml"
        if os.path.exists(base_path):
            base_conf = OmegaConf.load(base_path)
        else:
            logger.warning(
                f"Base config not found at {base_path}. Starting with empty config.")
            base_conf = OmegaConf.create()

        # 2. User Configの読み込みとマージ
        if args.config and args.config != base_path:
            if os.path.exists(args.config):
                user_conf = OmegaConf.load(args.config)
                base_conf = OmegaConf.merge(base_conf, user_conf)
            else:
                logger.error(f"❌ User Config file not found: {args.config}")
                sys.exit(1)

        # 3. Model Configの読み込みとマージ
        model_config_path = args.model_config

        # [修正] プレースホルダー "..." またはファイルが存在しない場合の自動フォールバック処理
        should_fallback = False
        if model_config_path:
            if model_config_path == "..." or not os.path.exists(model_config_path):
                logger.warning(
                    f"⚠️ Model config file '{model_config_path}' not found (or is placeholder).")
                should_fallback = True
            else:
                # 正常に読み込み
                model_conf_raw = OmegaConf.load(model_config_path)
                if 'model' in model_conf_raw:
                    model_conf = model_conf_raw
                else:
                    model_conf = OmegaConf.create({"model": model_conf_raw})
                base_conf = OmegaConf.merge(base_conf, model_conf)
                logger.info(f"✅ Loaded model config from {model_config_path}")

        # 4. コマンドライン引数によるオーバーライド (先に適用してよい)
        if args.override_config:
            for override in args.override_config:
                base_conf = OmegaConf.merge(
                    base_conf, OmegaConf.from_dotlist([override]))

        # 5. その他の引数による更新
        if args.data_path:
            OmegaConf.update(base_conf, "data.path", args.data_path)
        if args.paradigm:
            OmegaConf.update(base_conf, "training.paradigm", args.paradigm)

        # 6. 型変換
        if not isinstance(base_conf, DictConfig):
            if isinstance(base_conf, ListConfig):
                raise TypeError(
                    "Root config must be a dictionary (DictConfig), not a list.")
            base_conf = OmegaConf.create(base_conf)

        base_conf_typed = cast(DictConfig, base_conf)

        # [修正] architecture_type が未定義の場合の最終フォールバック
        model_arch = OmegaConf.select(
            base_conf_typed, "model.architecture_type")
        if not model_arch:
            model_arch = OmegaConf.select(base_conf_typed, "architecture_type")

        if (not model_arch or model_arch == 'unknown') or should_fallback:
            # デフォルト設定ファイル (small.yaml) へのフォールバックを試行
            default_model_path = "configs/models/small.yaml"
            if os.path.exists(default_model_path):
                logger.warning(
                    f"🔄 Falling back to default model config: {default_model_path}")
                default_conf_raw = OmegaConf.load(default_model_path)
                if 'model' in default_conf_raw:
                    default_model_conf = default_conf_raw
                else:
                    default_model_conf = OmegaConf.create(
                        {"model": default_conf_raw})

                # ベース設定にマージ（ユーザー設定は優先されるべきだが、アーキテクチャが無い場合はこれで埋める）
                base_conf = OmegaConf.merge(base_conf, default_model_conf)
                # 再取得
                base_conf_typed = cast(DictConfig, base_conf)
                model_arch = OmegaConf.select(
                    base_conf_typed, "model.architecture_type")
            else:
                logger.error(
                    "❌ 'architecture_type' is missing and default config 'configs/models/small.yaml' was not found.")
                sys.exit(1)

        logger.info(f"🔍 Loaded Config: architecture_type = {model_arch}")

    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    injected_tokenizer: Optional[PreTrainedTokenizerBase] = None
    try:
        tokenizer_name = str(OmegaConf.select(
            base_conf_typed, "data.tokenizer_name", default="gpt2"))
        injected_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        logger.warning(
            "Could not load configured AutoTokenizer. Will fallback to 'gpt2' later if needed.")
        injected_tokenizer = None

    train(args, config=base_conf_typed, tokenizer=injected_tokenizer)


if __name__ == "__main__":
    main()
