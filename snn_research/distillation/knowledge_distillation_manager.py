# ファイルパス: snn_research/distillation/knowledge_distillation_manager.py
# 日本語タイトル: Knowledge Distillation Manager v2.1 (Async & Non-blocking)
# 目的・内容:
#   知識蒸留プロセスを管理するマネージャーの非同期対応強化版。
#   - 教師モデルのロードやデータセットの事前計算（ロジット抽出）といった重い処理を
#     スレッドプールにオフロードし、Brain Kernelのイベントループをブロックしないように修正。
#   - AsyncArtificialBrain からの呼び出しに完全対応。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Dict, Any, Optional, List, Callable, Tuple, cast, TypeAlias, Sized, Union
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from omegaconf import DictConfig

# 循環参照回避のため、datasetのインポートはtry-exceptでラップ
try:
    from snn_research.data.datasets import SimpleTextDataset
except ImportError:
    class SimpleTextDataset(Dataset): # type: ignore
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 0
        def __getitem__(self, idx): return {}

import torchvision.models as models # type: ignore[import-untyped]

from snn_research.distillation.model_registry import ModelRegistry
from snn_research.training.trainers import DistillationTrainer

logger = logging.getLogger(__name__)

TextCollateFnDef: TypeAlias = Callable[[PreTrainedTokenizerBase, bool], Callable[[List[Any]], Any]]

try:
    from app.utils import collate_fn as text_collate_fn
    collate_fn_orig_factory: TextCollateFnDef = cast(TextCollateFnDef, text_collate_fn)
except ImportError:
    logger.warning("Warning: Could not import collate_fn from app.utils.py.")
    def _fallback_collate(batch: List[Any]) -> Any:
        raise NotImplementedError("Fallback collate_fn called.")
    def fallback_collate_fn_def(tokenizer: PreTrainedTokenizerBase, is_distillation: bool) -> Callable[[List[Any]], Any]:
        return _fallback_collate
    collate_fn_orig_factory = fallback_collate_fn_def


class _PrecomputedDistillationDataset(Dataset):
    """
    初期化時に教師モデルのロジットを一括計算してキャッシュするDataset。
    """
    def __init__(
        self,
        original_dataset: Dataset,
        teacher_model: nn.Module,
        device: str,
        collate_fn_orig: Callable[[List[Any]], Any],
        batch_size: int = 16
    ):
        self.original_dataset = original_dataset
        self.cached_logits: List[torch.Tensor] = []
        
        logger.info(f"🔄 Pre-computing teacher logits for {len(cast(Sized, original_dataset))} samples...")
        
        teacher_model.eval()
        
        # 推論用のDataLoader (num_workers=0で現在のスレッドで実行)
        temp_loader = DataLoader(
            original_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_orig
        )
        
        # 推論ループ
        with torch.no_grad():
            for batch in temp_loader:
                teacher_logits: torch.Tensor
                
                if isinstance(batch, dict):
                    if 'input_ids' in batch:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch.get('attention_mask')
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device)
                        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                        teacher_logits = outputs.logits
                        
                    elif 'input_images' in batch:
                        input_images = batch['input_images'].to(device)
                        teacher_logits = teacher_model(input_images)
                        if teacher_logits.dim() == 2:
                            teacher_logits = teacher_logits.unsqueeze(1)
                    else:
                        continue
                elif isinstance(batch, (list, tuple)):
                     inputs = batch[0].to(device)
                     if inputs.dtype == torch.long: # text
                         outputs = teacher_model(input_ids=inputs)
                         teacher_logits = outputs.logits
                     else: # image
                         teacher_logits = teacher_model(inputs)
                else:
                    continue

                # メモリ節約のためCPUへ移動しFP16化
                teacher_logits_cpu = teacher_logits.cpu().to(torch.float16)
                
                for i in range(teacher_logits_cpu.size(0)):
                    self.cached_logits.append(teacher_logits_cpu[i])
        
        logger.info("✅ Teacher logits pre-computation complete.")

    def __len__(self) -> int:
        return len(self.cached_logits)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        return self.original_dataset[idx], self.cached_logits[idx]


class KnowledgeDistillationManager:
    def __init__(
        self,
        student_model: nn.Module,
        trainer: DistillationTrainer,
        model_registry: ModelRegistry,
        device: str,
        config: DictConfig,
        teacher_model: Optional[nn.Module] = None,
        teacher_model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_model_name = teacher_model_name
        self.trainer = trainer
        self.model_registry = model_registry
        self.device = device
        self.config = config 
        
        # 重い処理用にExecutorを用意
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DistillationWorker")

        if not teacher_model and not teacher_model_name:
            raise ValueError("Either teacher_model (instance) or teacher_model_name (str) must be provided.")
            
        self.tokenizer_name = tokenizer_name if tokenizer_name else cast(str, teacher_model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception:
            logger.warning(f"Could not load tokenizer '{self.tokenizer_name}'. Using gpt2 fallback.")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_teacher_model_sync(self) -> nn.Module:
        """同期的に教師モデルをロードする（別スレッドで実行用）"""
        if self.teacher_model:
            return self.teacher_model.to(self.device).eval()

        if not self.teacher_model_name:
             raise ValueError("teacher_model_name is not set.")

        print(f"🧠 Loading teacher model '{self.teacher_model_name}' on {self.device}...")
        try:
            if self.teacher_model_name == "resnet18":
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                model.fc = torch.nn.Linear(model.fc.in_features, 10)
                teacher_weights_path = f"models/{self.teacher_model_name}_cifar10.pth"
                if os.path.exists(teacher_weights_path):
                    state_dict = torch.load(teacher_weights_path, map_location=self.device)
                    model.load_state_dict(state_dict)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.teacher_model_name)
            
            model = model.to(self.device).eval()
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load teacher model: {e}")
            raise

    async def _get_or_load_teacher_model(self) -> nn.Module:
        """非同期ラッパー: 教師モデルを取得またはロード"""
        if self.teacher_model:
            return self.teacher_model

        loop = asyncio.get_running_loop()
        # 重いモデルロードを別スレッドで実行
        self.teacher_model = await loop.run_in_executor(
            self.executor, self._load_teacher_model_sync
        )
        return cast(nn.Module, self.teacher_model)

    async def run_on_demand_pipeline(
        self,
        task_description: str,
        unlabeled_data_path: str,
        force_retrain: bool = False,
        student_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        print(f"--- On-Demand Learning Pipeline Initiated: {task_description} ---")

        # 1. 既存モデル検索
        if not force_retrain:
            existing_experts = await self.model_registry.find_models_for_task(task_description, top_k=1)
            if existing_experts:
                best_expert = existing_experts[0]
                best_expert['model_id'] = task_description
                print(f"✅ Found existing expert: {best_expert.get('model_path')}")
                return best_expert

        if not os.path.exists(unlabeled_data_path):
            return {"error": f"Data file not found: {unlabeled_data_path}"}
        
        # 2. データセット準備とロジット計算
        try:
            # Dataset初期化は軽いのでここで実行
            train_dataset_raw: Dataset = SimpleTextDataset(
                file_path=unlabeled_data_path,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.model.time_steps 
            )
            
            if len(cast(Sized, train_dataset_raw)) < 10:
                 if len(cast(Sized, train_dataset_raw)) == 0:
                     return {"error": "No data found."}
                 train_dataset_raw = torch.utils.data.ConcatDataset([train_dataset_raw] * (10 // len(cast(Sized, train_dataset_raw)) + 1))

            print("Preparing distillation dataset (This may take a while, but Brain is active)...")
            
            # ここが重い処理：非同期で実行
            train_loader, val_loader = await self.prepare_dataset( 
                train_dataset_raw,
                None, 
                batch_size=self.config.training.batch_size, 
                collate_fn=None 
            )

        except Exception as e:
            logger.error(f"❌ Dataset preparation failed: {e}", exc_info=True)
            return {"error": str(e)}

        print(f"Starting distillation training for {self.config.training.epochs} epochs...")
        
        # 3. 学習実行
        # Trainer内部も同期的なので、全体をラップするか、Trainerを非同期化する必要がある
        # 現状は簡易的にここもExecutorで回す（ただしGPU操作を含むため注意が必要）
        # PyTorchのGPU操作はGILを解放するため、標準のスレッドプールでも並行性は出る
        loop = asyncio.get_running_loop()
        final_metrics = await loop.run_in_executor(
            self.executor,
            partial(
                self._run_distillation_sync,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config.training.epochs,
                model_id=task_description,
                task_description=task_description,
                student_config=student_config
            )
        )

        print(f"✅ On-demand learning finished.")
        return final_metrics

    def _run_distillation_sync(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        model_id: str,
        task_description: str,
        student_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """同期実行用の学習ループ（run_in_executorから呼ばれる）"""
        best_metric = float('inf') 
        best_model_path = ""
        log_dir = self.config.training.log_dir 
        os.makedirs(log_dir, exist_ok=True)

        for epoch in range(epochs):
            # 学習
            _ = self.trainer.train_epoch(train_loader, epoch)
            
            # 評価
            if val_loader:
                val_metrics = self.trainer.evaluate(val_loader, epoch)
                metric_name = self.config.training.get("metric_to_optimize", "total")
                current_metric = val_metrics.get(metric_name, float('inf'))

                if current_metric < best_metric:
                    best_metric = current_metric
                    safe_model_id = model_id.replace(" ", "_").replace("/", "_")
                    best_model_path = os.path.join(log_dir, f"{safe_model_id}_best.pth")
                    config_to_save = student_config if student_config is not None else {} 
                    
                    self.trainer.save_checkpoint(
                        path=best_model_path,
                        epoch=epoch,
                        metric_value=best_metric,
                        config=config_to_save, 
                        tokenizer_name=self.tokenizer_name
                    )

        # 最終結果作成
        final_metrics: Dict[str, Any] = {"accuracy": 0.0}
        if val_loader and os.path.exists(best_model_path):
            self.trainer.load_checkpoint(best_model_path)
            eval_res = self.trainer.evaluate(val_loader, epochs)
            final_metrics['accuracy'] = eval_res.get('accuracy', 0.0)

        # レジストリ登録は非同期メソッドなので、ここでは行わずパスを返すのが安全だが、
        # Registryがasyncio.runを内部で使うか、ここだけ同期版メソッドがあれば呼ぶ。
        # ここでは簡易的に情報を返すにとどめる。
        
        return {
            "metrics": final_metrics,
            "path": best_model_path,
            "status": "training_complete"
        }

    def _create_dataset_sync(
        self,
        dataset: Dataset,
        teacher_model: nn.Module,
        collate_fn: Callable,
        batch_size: int
    ) -> Dataset:
        """データセット作成の同期処理"""
        return _PrecomputedDistillationDataset(
            original_dataset=dataset,
            teacher_model=teacher_model,
            device=self.device,
            collate_fn_orig=collate_fn,
            batch_size=batch_size
        )

    async def prepare_dataset(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None
    ) -> Tuple[DataLoader, DataLoader]:
        
        # Collate準備
        collate_fn_to_use: Callable[[List[Any]], Any]
        if collate_fn is None:
            collate_fn_factory = cast(TextCollateFnDef, collate_fn_orig_factory)
            collate_fn_to_use = collate_fn_factory(self.tokenizer, False)
        else:
            collate_fn_to_use = collate_fn

        # 教師モデルロード（非同期）
        teacher_model_instance = await self._get_or_load_teacher_model()

        loop = asyncio.get_running_loop()
        
        # トレーニングデータのロジット計算（別スレッド）
        print("   -> Computing logits for Training set...")
        distill_train_dataset = await loop.run_in_executor(
            self.executor,
            partial(
                self._create_dataset_sync,
                dataset=train_dataset,
                teacher_model=teacher_model_instance,
                collate_fn=collate_fn_to_use,
                batch_size=batch_size
            )
        )
        
        # 検証データのロジット計算（別スレッド）
        distill_val_dataset: Dataset
        if val_dataset:
            print("   -> Computing logits for Validation set...")
            distill_val_dataset = await loop.run_in_executor(
                self.executor,
                partial(
                    self._create_dataset_sync,
                    dataset=val_dataset,
                    teacher_model=teacher_model_instance,
                    collate_fn=collate_fn_to_use,
                    batch_size=batch_size
                )
            )
        else:
            # 簡易分割
            train_len = len(cast(Sized, distill_train_dataset))
            train_size = int(0.9 * train_len)
            val_size = train_len - train_size
            if train_size > 0 and val_size > 0:
                distill_train_dataset, distill_val_dataset = torch.utils.data.random_split(distill_train_dataset, [train_size, val_size])
            else:
                distill_val_dataset = distill_train_dataset

        # Collate関数作成
        distillation_collate_fn = self._create_distillation_collate_fn(collate_fn_to_use)

        # DataLoader作成
        train_loader = DataLoader(
            distill_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=distillation_collate_fn
        )
        val_loader = DataLoader(
            distill_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=distillation_collate_fn
        )

        return train_loader, val_loader

    def _create_distillation_collate_fn(self, collate_fn_orig: Callable[[List[Any]], Any]) -> Callable:
        """蒸留用Collate関数（変更なし）"""
        def distillation_collate(batch: List[Tuple[Dict[str, Any], torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            original_batch_items = [item[0] for item in batch]
            teacher_logits_list = [item[1] for item in batch]

            collated_batch = collate_fn_orig(original_batch_items)
            
            if 'input_ids' in collated_batch:
                student_input = collated_batch['input_ids']
                attention_mask = collated_batch['attention_mask']
                student_target = collated_batch['labels']
            elif 'input_images' in collated_batch:
                student_input = collated_batch['input_images'] 
                student_target = collated_batch['labels']      
                attention_mask = torch.ones_like(student_target, dtype=torch.long) 
            else:
                raise KeyError("Unknown batch format")

            padded_teacher_logits = torch.nn.utils.rnn.pad_sequence(
                teacher_logits_list, batch_first=True, padding_value=0.0
            )

            # 形状調整
            if student_input.dim() > 2: # Image
                if padded_teacher_logits.dim() == 3 and padded_teacher_logits.shape[1] == 1:
                    padded_teacher_logits = padded_teacher_logits.squeeze(1)
                return student_input, attention_mask, student_target, padded_teacher_logits

            # Text Length Matching
            max_len_student = student_input.shape[1]
            max_len_teacher = padded_teacher_logits.shape[1]
            
            if max_len_student > max_len_teacher:
                pad_size = max_len_student - max_len_teacher
                padding = torch.zeros(
                    (padded_teacher_logits.shape[0], pad_size, padded_teacher_logits.shape[2]),
                    dtype=padded_teacher_logits.dtype, device=padded_teacher_logits.device
                )
                padded_teacher_logits = torch.cat([padded_teacher_logits, padding], dim=1)
            elif max_len_teacher > max_len_student:
                pad_size = max_len_teacher - max_len_student
                pad_val = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padding_input = torch.full((student_input.shape[0], pad_size), pad_val, dtype=student_input.dtype, device=student_input.device)
                student_input = torch.cat([student_input, padding_input], dim=1)
                padding_mask = torch.zeros((attention_mask.shape[0], pad_size), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding_mask], dim=1)
                padding_target = torch.full((student_target.shape[0], pad_size), -100, dtype=student_target.dtype, device=student_target.device)
                student_target = torch.cat([student_target, padding_target], dim=1)
            
            return student_input, attention_mask, student_target, padded_teacher_logits

        return distillation_collate