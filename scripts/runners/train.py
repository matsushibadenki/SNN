# ファイルパス: scripts/runners/train.py
# (修正: mypyエラー修正 - tokenizerのNoneチェックと型ナローイングを追加)

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler, Dataset, Sampler
from dependency_injector.wiring import inject, Provide
from typing import Optional, Tuple, List, Dict, Any, Callable, cast, Union, TYPE_CHECKING
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from app.containers import TrainingContainer
from snn_research.data.datasets import get_dataset_class, DistillationDataset, DataFormat, SNNBaseDataset
from snn_research.training.trainers import BreakthroughTrainer, ParticleFilterTrainer, DistillationTrainer
from snn_research.training.bio_trainer import BioRLTrainer 
from snn_research.training.quantization import apply_qat, convert_to_quantized_model, apply_spquant_quantization
from snn_research.training.pruning import apply_sbc_pruning, apply_spatio_temporal_pruning
from scripts.data_preparation import prepare_wikitext_data
from snn_research.core.snn_core import SNNCore
from app.utils import get_auto_device, collate_fn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train(args: argparse.Namespace, config: DictConfig, tokenizer: Optional[PreTrainedTokenizerBase]) -> None:
    # --- コンテナをここで新規作成 ---
    container = TrainingContainer()
    
    if config:
        # Configを辞書に変換して適用。戻り値を明示的にキャスト
        conf_dict = cast(Dict[str, Any], OmegaConf.to_container(config, resolve=True))
        
        if isinstance(conf_dict, dict):
            container.config.from_dict(conf_dict)
            
            # デバッグ: モデル設定が正しく渡っているか確認
            model_conf = conf_dict.get('model', {})
            if isinstance(model_conf, dict):
                arch = model_conf.get('architecture_type', 'NOT_FOUND')
            else:
                arch = 'INVALID_MODEL_CONFIG'

            logger.info(f"🔧 Train Config Loaded. Architecture: {arch}")
            if arch == 'NOT_FOUND':
                logger.warning(f"⚠️ Model config looks empty! Keys: {conf_dict.keys()}")
        else:
            logger.error(f"❌ Config is not a dictionary, got {type(conf_dict)}")
            return
    # -------------------------------

    is_distributed = args.distributed
    rank = int(os.environ.get("LOCAL_RANK", -1))
    device = f'cuda:{rank}' if is_distributed and torch.cuda.is_available() else get_auto_device()
    
    paradigm = config.training.paradigm
    logger.info(f"🚀 学習パラダイム: {paradigm}")

    trainer: Union[BreakthroughTrainer, BioRLTrainer, ParticleFilterTrainer]

    if paradigm.startswith("bio-"):
        if paradigm == "bio-causal-sparse":
            container.config.training.biologically_plausible.adaptive_causal_sparsification.enabled.from_value(True)
            trainer = container.bio_rl_trainer()
            cast(BioRLTrainer, trainer).train(num_episodes=config.training.epochs)
        elif paradigm == "bio-particle-filter":
            trainer = container.particle_filter_trainer()
            dummy_data = torch.rand(1, 10, device=device)
            dummy_targets = torch.rand(1, 2, device=device)
            for epoch in range(config.training.epochs):
                loss = cast(ParticleFilterTrainer, trainer).train_step(dummy_data, dummy_targets)
                logger.info(f"Epoch {epoch+1}/{config.training.epochs}: Particle Filter Loss = {loss:.4f}")
        elif paradigm == "bio-probabilistic-hebbian":
            prob_trainer: BioRLTrainer = container.probabilistic_trainer()
            prob_trainer.train(num_episodes=config.training.epochs)
        else:
            raise ValueError(f"Unknown bio paradigm: {paradigm}")

    elif paradigm in ["gradient_based", "self_supervised", "physics_informed", "probabilistic_ensemble"]:
        grad_config = config.training.get("gradient_based", {})
        grad_type = grad_config.get("type", "standard")
        
        is_distillation = (paradigm == "gradient_based" and grad_type == "distillation")
        logger.info(f"ℹ️ Gradient Training Type: {grad_type} (Is Distillation: {is_distillation})")

        if args.data_path:
            data_path = args.data_path
        else:
            data_path = OmegaConf.select(config, "data.path", default="data/default_data.jsonl")
            if data_path == "data/wikitext-103_train.jsonl" and not os.path.exists(data_path):
                 prepare_wikitext_data()
        
        if not os.path.exists(data_path):
             if "smoke_test_data.jsonl" in data_path:
                 logger.info(f"⚠️ Data not found: {data_path}. Creating dummy data.")
                 os.makedirs(os.path.dirname(data_path), exist_ok=True)
                 with open(data_path, 'w') as f:
                     import json
                     for i in range(10): f.write(json.dumps({"text": f"This is a smoke test sample {i}."}) + "\n")
             else:
                 raise FileNotFoundError(f"Data file not found: {data_path}")

        DatasetClass = get_dataset_class(DataFormat(config.data.format))
        max_seq_len = OmegaConf.select(config, "model.time_steps", default=128)
        dataset: SNNBaseDataset

        # --- ▼ 修正: Tokenizerの安全確保 ▼ ---
        if tokenizer is None:
            logger.warning("Tokenizer is None. Attempting to load default 'gpt2' tokenizer.")
            try:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except Exception as e:
                raise ValueError("Tokenizer is required for dataset initialization but could not be loaded.") from e
        
        # 型ナローイング: これ以降 tokenizer は None ではないと扱われる
        assert tokenizer is not None
        # --- ▲ 修正 ▲ ---

        if is_distillation:
            data_dir = os.path.dirname(data_path) if os.path.isfile(data_path) else data_path
            distill_jsonl_path = os.path.join(data_dir, "distillation_data.jsonl")
            if isinstance(DatasetClass, type(DistillationDataset)) or os.path.exists(distill_jsonl_path):
                 if os.path.exists(distill_jsonl_path):
                     logger.info(f"Using DistillationDataset from {distill_jsonl_path}")
                     dataset = DistillationDataset(file_path=distill_jsonl_path, data_dir=data_dir, tokenizer=tokenizer, max_seq_len=max_seq_len)
                 else:
                     logger.warning("⚠️ 蒸留モードですが、蒸留用データセットが見つかりません。標準学習モードに切り替えます。")
                     is_distillation = False 
                     dataset = DatasetClass(file_path=data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)
            else:
                 logger.warning("⚠️ 蒸留モードですが、適切なデータセットがありません。標準学習モードに切り替えます。")
                 is_distillation = False
                 dataset = DatasetClass(file_path=data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)
        else:
            dataset = DatasetClass(file_path=data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)
        
        logger.info(f"Dataset loaded: {len(dataset)} samples. Final Is Distillation: {is_distillation}")

        split_ratio = OmegaConf.select(config, "data.split_ratio", default=0.1)
        train_size = int((1.0 - split_ratio) * len(dataset))
        val_size = len(dataset) - train_size
        if train_size <= 0: train_size = len(dataset); val_size = 0
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_sampler: Optional[DistributedSampler] = DistributedSampler(train_dataset) if is_distributed else None
        
        collate_fn_instance = collate_fn(tokenizer, is_distillation)
        
        train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn_instance)
        val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn_instance)

        snn_model: nn.Module = container.snn_model(backend=args.backend)
        snn_model.to(device)
        if is_distributed: snn_model = DDP(snn_model, device_ids=[rank], find_unused_parameters=True)

        astrocyte = container.astrocyte_network(snn_model=snn_model) if args.use_astrocyte else None
        optimizer = container.optimizer(params=snn_model.parameters())
        scheduler = container.scheduler(optimizer=optimizer) if config.training.gradient_based.use_scheduler else None
        
        if paradigm == "gradient_based":
            if is_distillation:
                logger.info("🎓 Using DistillationTrainer")
                trainer = container.distillation_trainer(model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank)
            else:
                logger.info("👨‍🏫 Using Standard BreakthroughTrainer")
                trainer = container.standard_trainer(model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank)
        elif paradigm == "self_supervised":
            trainer = container.self_supervised_trainer(model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank)
        elif paradigm == "physics_informed":
            trainer = container.physics_informed_trainer(model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank)
        else: 
            trainer = container.probabilistic_ensemble_trainer(model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank)

        bt_trainer = cast(BreakthroughTrainer, trainer)
        if args.load_ewc_data: bt_trainer.load_ewc_data(args.load_ewc_data)

        start_epoch = bt_trainer.load_checkpoint(args.resume_path) if args.resume_path else 0
        for epoch in range(start_epoch, config.training.epochs):
            bt_trainer.train_epoch(train_loader, epoch)
            if rank in [-1, 0] and (epoch % config.training.eval_interval == 0 or epoch == config.training.epochs - 1):
                val_metrics = bt_trainer.evaluate(val_loader, epoch)
                if epoch % config.training.log_interval == 0:
                    checkpoint_path = os.path.join(config.training.log_dir, f"checkpoint_epoch_{epoch}.pth")
                    model_config_dict = cast(Dict[str, Any], OmegaConf.to_container(config.model, resolve=True)) if isinstance(config.model, DictConfig) else config.model
                    bt_trainer.save_checkpoint(path=checkpoint_path, epoch=epoch, metric_value=val_metrics.get('total', float('inf')), tokenizer_name=config.data.tokenizer_name, config=model_config_dict)
        
        if args.task_name and rank in [-1, 0]:
             ewc_weight = OmegaConf.select(config, "training.gradient_based.loss.ewc_weight", default=0.0)
             if ewc_weight > 0:
                 logger.info(f"🔒 Computing EWC Fisher Matrix for task '{args.task_name}'...")
                 bt_trainer._compute_ewc_fisher_matrix(train_loader, args.task_name)

    else:
        raise ValueError(f"Unknown training paradigm: '{paradigm}'.")

    logger.info("✅ 学習が完了しました。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/templates/base_config.yaml")
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--override_config", type=str, action='append')
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--resume_path", type=str)
    parser.add_argument("--load_ewc_data", type=str)
    parser.add_argument("--use_astrocyte", action="store_true")
    parser.add_argument("--paradigm", type=str)
    parser.add_argument("--backend", type=str, default="spikingjelly")
    args = parser.parse_args()

    try:
        base_path = "configs/templates/base_config.yaml"
        if os.path.exists(base_path):
            base_conf = OmegaConf.load(base_path)
        else:
            base_conf = OmegaConf.create()

        if args.config and args.config != base_path:
            if os.path.exists(args.config):
                user_conf = OmegaConf.load(args.config)
                base_conf = OmegaConf.merge(base_conf, user_conf)
            else:
                logger.warning(f"Config file not found: {args.config}")
        
        if args.model_config:
            if os.path.exists(args.model_config):
                model_conf_raw = OmegaConf.load(args.model_config)
                if 'model' in model_conf_raw:
                    model_conf = model_conf_raw
                else:
                    model_conf = OmegaConf.create({"model": model_conf_raw})
                base_conf = OmegaConf.merge(base_conf, model_conf)
        
        if args.override_config:
            for override in args.override_config:
                base_conf = OmegaConf.merge(base_conf, OmegaConf.from_dotlist([override]))
        
        if args.data_path:
            OmegaConf.update(base_conf, "data.path", args.data_path)
        if args.paradigm:
            OmegaConf.update(base_conf, "training.paradigm", args.paradigm)

        model_arch = OmegaConf.select(base_conf, "model.architecture_type")
        logger.info(f"🔍 Loaded Model Config: architecture_type = {model_arch}")

        final_config_dict = cast(Dict[str, Any], OmegaConf.to_container(base_conf, resolve=True))
        if not isinstance(final_config_dict, dict):
             raise TypeError("Final config is not a dictionary.")
        
        if not isinstance(base_conf, DictConfig):
            if isinstance(base_conf, ListConfig):
                 raise TypeError("Root config must be a dictionary (DictConfig), not a list (ListConfig).")
            else:
                 base_conf = OmegaConf.create(base_conf)
        
        base_conf_typed = cast(DictConfig, base_conf)

    except Exception as e:
        logger.error(f"Error loading/merging config: {e}")
        sys.exit(1)
    
    try:
        injected_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        logger.warning("Could not load AutoTokenizer, using simple fallback.")
        injected_tokenizer = None

    train(args, config=base_conf_typed, tokenizer=injected_tokenizer)

if __name__ == "__main__":
    main()