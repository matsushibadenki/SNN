# ファイルパス: scripts/runners/train.py

import sys
import os

# ------------------------------------------------------------------------------
# [Auto-inserted by fix_script_paths.py]
# プロジェクトルートディレクトリをsys.pathに追加して、snn_researchモジュールを解決可能にする
# このファイルは scripts/runners/ に配置されていることを想定しています (ルートから2階層下)
# ------------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------------------

# ファイルパス: train.py
# (修正: 設定ロード確認ログの追加)

import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
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
from snn_research.training.bio_trainer import BioRLTrainer # Correct import
from snn_research.training.quantization import apply_qat, convert_to_quantized_model, apply_spquant_quantization
from snn_research.training.pruning import apply_sbc_pruning, apply_spatio_temporal_pruning
from scripts.data_preparation import prepare_wikitext_data
from snn_research.core.snn_core import SNNCore
from app.utils import get_auto_device
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

container = TrainingContainer()

def collate_fn(tokenizer: PreTrainedTokenizerBase, is_distillation: bool) -> Callable[[List[Any]], Any]:
    def collate(batch: List[Any]) -> Any:
        padding_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        inputs: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []

        for item in batch:
            if isinstance(item, dict):
                inp = item.get('input_ids')
                tgt = item.get('labels')
                if inp is None or tgt is None: continue
                inputs.append(torch.tensor(inp) if not isinstance(inp, torch.Tensor) else inp)
                targets.append(torch.tensor(tgt) if not isinstance(tgt, torch.Tensor) else tgt)
                if is_distillation:
                    lg = item.get('teacher_logits')
                    if lg is not None: logits.append(torch.tensor(lg) if not isinstance(lg, torch.Tensor) else lg)
                    else: logits.append(torch.empty(0))

            elif isinstance(item, tuple) and len(item) >= 2:
                inp = item[0]
                tgt = item[1]
                inputs.append(torch.tensor(inp) if not isinstance(inp, torch.Tensor) else inp)
                targets.append(torch.tensor(tgt) if not isinstance(tgt, torch.Tensor) else tgt)
                if is_distillation:
                    if len(item) >= 3:
                         lg = item[2]
                         if lg is not None: logits.append(torch.tensor(lg) if not isinstance(lg, torch.Tensor) else lg)
                         else: logits.append(torch.empty(0))
                    else: logits.append(torch.empty(0))

        if not inputs or not targets:
            if is_distillation:
                return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0, 0), dtype=torch.float32)
            else:
                return {
                    "input_ids": torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_val),
                    "attention_mask": torch.nn.utils.rnn.pad_sequence([torch.ones_like(i) for i in inputs], batch_first=True, padding_value=0),
                    "labels": torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
                }

        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_val)
        padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
        attention_mask = torch.ones_like(padded_inputs)
        attention_mask[padded_inputs == padding_val] = 0
        
        if not is_distillation:
            return {
                "input_ids": padded_inputs,
                "attention_mask": attention_mask,
                "labels": padded_targets
            }
        
        padded_logits = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=0.0)
        
        seq_len = padded_inputs.shape[1]
        if padded_targets.shape[1] < seq_len:
            pad = torch.full((padded_targets.shape[0], seq_len - padded_targets.shape[1]), -100, dtype=padded_targets.dtype, device=padded_targets.device)
            padded_targets = torch.cat([padded_targets, pad], dim=1)
        if padded_logits.shape[1] < seq_len:
            pad = torch.full((padded_logits.shape[0], seq_len - padded_logits.shape[1], padded_logits.shape[2]), 0.0, dtype=padded_logits.dtype, device=padded_logits.device)
            padded_logits = torch.cat([padded_logits, pad], dim=1)
            
        return padded_inputs, attention_mask, padded_targets, padded_logits
    
    return collate


def train(args: argparse.Namespace, config: DictConfig, tokenizer: PreTrainedTokenizerBase) -> None:
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
                    model_config_dict = OmegaConf.to_container(config.model, resolve=True) if isinstance(config.model, DictConfig) else config.model
                    bt_trainer.save_checkpoint(path=checkpoint_path, epoch=epoch, metric_value=val_metrics.get('total', float('inf')), tokenizer_name=config.data.tokenizer_name, config=model_config_dict)
        
        if args.task_name and rank in [-1, 0]:
             # 継続学習のために、このタスクでの重要度（Fisher行列）を計算して保存
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

        # --- 追加: モデル設定のロード確認 ---
        model_arch = OmegaConf.select(base_conf, "model.architecture_type")
        logger.info(f"🔍 Loaded Model Config: architecture_type = {model_arch}")
        # -----------------------------------

        final_config_dict = cast(Dict[str, Any], OmegaConf.to_container(base_conf, resolve=True))
        if not isinstance(final_config_dict, dict):
             raise TypeError("Final config is not a dictionary.")
        
        container.config.from_dict(final_config_dict)
        
        # Ensure base_conf is a DictConfig before passing it to train
        if not isinstance(base_conf, DictConfig):
            if isinstance(base_conf, ListConfig):
                 raise TypeError("Root config must be a dictionary (DictConfig), not a list (ListConfig).")
            else:
                 base_conf = OmegaConf.create(base_conf)
        
        base_conf_typed = cast(DictConfig, base_conf)

    except Exception as e:
        logger.error(f"Error loading/merging config: {e}")
        sys.exit(1)

    container.wire(modules=[__name__])
    
    try:
        injected_tokenizer = container.tokenizer()
    except Exception:
        injected_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    train(args, config=base_conf_typed, tokenizer=injected_tokenizer)

if __name__ == "__main__":
    main()