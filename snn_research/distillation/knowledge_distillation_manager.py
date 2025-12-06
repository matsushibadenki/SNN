# ファイルパス: snn_research/distillation/knowledge_distillation_manager.py
# (修正版: インポート整理とデータセット処理の改善)
# Title: 知識蒸留 (Knowledge Distillation) 管理マネージャー
# Description:
# - 教師モデルからの知識蒸留プロセスを管理するクラス。
# - オンデマンド学習パイプライン (run_on_demand_pipeline) を提供。
# - 教師モデルのロジットを事前計算する _PrecomputedDistillationDataset を導入し、
#   DataLoader のマルチプロセス動作時のエラー回避と高速化を実現。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Dict, Any, Optional, List, Callable, Tuple, cast, TypeAlias, Sized, Union
import os
import json
import logging
import asyncio
from omegaconf import DictConfig

# snn_research.data.datasets から SimpleTextDataset をインポート
# 循環参照のリスクを避けるため、インポートエラー時はダミーを使用（型チェック用）
try:
    from snn_research.data.datasets import SimpleTextDataset
except ImportError:
    # テストや環境によってはインポートできない場合のフォールバック
    class SimpleTextDataset(Dataset): # type: ignore
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 0
        def __getitem__(self, idx): return {}

# type: ignore[import-untyped]
import torchvision.models as models 

from snn_research.distillation.model_registry import ModelRegistry
from snn_research.training.trainers import DistillationTrainer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)

# 型エイリアスの定義
TextCollateFnDef: TypeAlias = Callable[[PreTrainedTokenizerBase, bool], Callable[[List[Any]], Any]]

# app.utils から collate_fn をインポート
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
    DataLoader の worker プロセス内でのモデル推論（CUDA初期化エラーの要因）を回避する。
    """
    def __init__(
        self,
        original_dataset: Dataset,
        teacher_model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        collate_fn_orig: Callable[[List[Any]], Any],
        device: str,
        batch_size: int = 16
    ):
        self.original_dataset = original_dataset
        # データセット自体はロジットのみを保持し、モデルは保持しない（Pickle対策）
        self.cached_logits: List[torch.Tensor] = []
        
        logger.info(f"Pre-computing teacher logits for {len(cast(Sized, original_dataset))} samples...")
        
        teacher_model.eval()
        
        # 推論用のDataLoaderを作成（num_workers=0 でメインプロセス実行）
        temp_loader = DataLoader(
            original_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_orig
        )
        
        with torch.no_grad():
            for batch in temp_loader:
                teacher_logits: torch.Tensor
                
                if isinstance(batch, dict):
                    if 'input_ids' in batch:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch.get('attention_mask')
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device)
                        
                        # 教師モデルで推論
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
                     # (inputs, targets) 形式の場合のフォールバック
                     # テキスト分類などを想定
                     inputs = batch[0].to(device)
                     if inputs.dtype == torch.long: # text
                         outputs = teacher_model(input_ids=inputs)
                         teacher_logits = outputs.logits
                     else: # image
                         teacher_logits = teacher_model(inputs)
                else:
                    continue

                # ロジットをCPUに移動し、FP16にしてメモリ節約
                teacher_logits_cpu = teacher_logits.cpu().to(torch.float16)
                
                # バッチ次元で分割してリストに追加
                for i in range(teacher_logits_cpu.size(0)):
                    self.cached_logits.append(teacher_logits_cpu[i])
        
        logger.info("✅ Teacher logits pre-computation complete.")

    def __len__(self) -> int:
        return len(self.cached_logits)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        # 元のデータ項目と、事前計算済みのロジットを返す
        original_item = self.original_dataset[idx]
        cached_logit = self.cached_logits[idx]
        return original_item, cached_logit


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

        if not teacher_model and not teacher_model_name:
            raise ValueError("Either teacher_model (instance) or teacher_model_name (str) must be provided.")
            
        if not tokenizer_name and not (isinstance(teacher_model_name, str) and teacher_model_name):
             raise ValueError("tokenizer_name or a valid teacher_model_name must be provided to load tokenizer.")

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

    async def _get_or_load_teacher_model(self) -> nn.Module:
        """
        教師モデルのインスタンスを取得する。
        """
        if self.teacher_model:
            return self.teacher_model.to(self.device).eval()

        if not self.teacher_model_name:
             raise ValueError("Cannot load teacher model: teacher_model_name is not set.")

        print(f"🧠 Loading teacher model '{self.teacher_model_name}'...")
        try:
            if self.teacher_model_name == "resnet18":
                # ImageNetの重みで初期化
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                num_ftrs = model.fc.in_features
                # 分類ヘッドをCIFAR-10用に変更 (10クラス)
                model.fc = torch.nn.Linear(num_ftrs, 10) 
                
                # --- タスク特化の学習済み重みがあればロードする ---
                teacher_weights_path = f"models/{self.teacher_model_name}_cifar10.pth"
                if os.path.exists(teacher_weights_path):
                    print(f"   -> Loading fine-tuned weights from {teacher_weights_path}")
                    state_dict = torch.load(teacher_weights_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                else:
                    print(f"⚠️ Warning: Fine-tuned weights not found at '{teacher_weights_path}'.")
                    print("   -> The teacher model's classification head is randomly initialized!")
                    print("   -> Distillation efficiency will be extremely low.")
                
            else:
                model = AutoModelForCausalLM.from_pretrained(self.teacher_model_name)
            
            self.teacher_model = model.to(self.device).eval()
            return self.teacher_model
        except Exception as e:
            print(f"❌ Failed to load teacher model: {e}")
            raise

    async def run_on_demand_pipeline(
        self,
        task_description: str,
        unlabeled_data_path: str,
        force_retrain: bool = False,
        student_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        print(f"--- On-Demand Learning Pipeline Initiated ---")
        print(f"Task: {task_description}")

        # 1. 既存モデルの検索
        if not force_retrain:
            existing_experts = await self.model_registry.find_models_for_task(task_description, top_k=1)
            if existing_experts:
                best_expert = existing_experts[0]
                best_expert['model_id'] = task_description # タスクIDとして使用
                print(f"✅ Found existing expert: {best_expert.get('model_path')}")
                return best_expert

        print(f"ℹ️ No suitable expert found or retraining forced. Starting new training.")

        if not os.path.exists(unlabeled_data_path):
            print(f"❌ Error: Unlabeled data file not found at '{unlabeled_data_path}'")
            return {"error": "Data file not found"}
        
        # 2. データセットの準備
        try:
            # SimpleTextDataset はトップレベルでインポート済み
            train_dataset_raw: Dataset = SimpleTextDataset(
                file_path=unlabeled_data_path,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.model.time_steps 
            )
            
            # データが少なすぎる場合の増量処理
            if len(cast(Sized, train_dataset_raw)) < 10:
                 print(f"⚠️ Warning: Dataset at '{unlabeled_data_path}' is too small.")
                 if len(cast(Sized, train_dataset_raw)) == 0:
                     return {"error": "No data found in the provided file."}
                 # データセットを複製して増やす (ConcatDatasetを使用)
                 train_dataset_raw = torch.utils.data.ConcatDataset([train_dataset_raw] * (10 // len(cast(Sized, train_dataset_raw)) + 1))

            print("Preparing distillation dataset (pre-calculating teacher logits)...")
            
            train_loader, val_loader = await self.prepare_dataset( 
                train_dataset_raw,
                None, 
                batch_size=self.config.training.batch_size, 
                collate_fn=None 
            )

        except Exception as e:
            print(f"❌ Error preparing dataset: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Dataset preparation failed: {e}"}

        print(f"Starting distillation training for {self.config.training.epochs} epochs...")
        
        # 3. 蒸留学習の実行
        final_metrics: Dict[str, Any] = await self.run_distillation( 
            train_loader=train_loader,
            val_loader=val_loader, 
            epochs=self.config.training.epochs, 
            model_id=task_description, 
            task_description=task_description,
            student_config=student_config 
        )

        print(f"✅ On-demand learning finished.")
        return final_metrics

    async def run_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        model_id: str,
        task_description: str,
        student_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        best_metric = float('inf') 
        best_model_path = ""
        
        log_dir = self.config.training.log_dir 
        os.makedirs(log_dir, exist_ok=True)

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            # トレーナーによる学習（1エポック）
            train_metrics = self.trainer.train_epoch(train_loader, epoch)
            
            # 評価とチェックポイント保存
            if val_loader:
                val_metrics = self.trainer.evaluate(val_loader, epoch)
                metric_name = self.config.training.get("metric_to_optimize", "total")
                current_metric = val_metrics.get(metric_name, float('inf'))

                print(f"Epoch {epoch + 1} Validation Metrics: {val_metrics}")

                if current_metric < best_metric:
                    best_metric = current_metric
                    # ファイル名にエージェントが指定したIDを使用
                    safe_model_id = model_id.replace(" ", "_").replace("/", "_")
                    best_model_path = os.path.join(log_dir, f"{safe_model_id}_best.pth")
                    config_to_save: Dict[str, Any] = student_config if student_config is not None else {} 
                    
                    self.trainer.save_checkpoint(
                        path=best_model_path,
                        epoch=epoch,
                        metric_value=best_metric,
                        config=config_to_save, 
                        tokenizer_name=self.tokenizer_name
                    )
            else:
                 best_metric = train_metrics.get("total", float('inf'))

        # 最終評価
        print("\n--- Final Evaluation on Validation Set ---")
        final_metrics: Dict[str, Any] = {"accuracy": 0.0, "avg_spikes_per_sample": float('inf')}
        
        if val_loader:
            if os.path.exists(best_model_path):
                # ベストモデルをロード
                self.trainer.load_checkpoint(best_model_path)
            
            final_eval_metrics_raw = self.trainer.evaluate(val_loader, epochs)
            final_metrics['accuracy'] = final_eval_metrics_raw.get('accuracy', 0.0) 
            final_metrics['avg_spikes_per_sample'] = final_eval_metrics_raw.get('avg_cutoff_steps', 0.0) 
            
        print(f"Final Metrics: {final_metrics}")

        # レジストリへの登録
        if student_config:
            await self.model_registry.register_model(
                model_id=model_id,
                task_description=task_description,
                metrics=final_metrics,
                model_path=best_model_path,
                config=student_config
            )
            final_model_info: Dict[str, Any] = { 
                "model_id": model_id,
                "task_description": task_description,
                "metrics": final_metrics,
                "path": best_model_path,
                "config": student_config
            }
            return final_model_info
        else:
            print("⚠️ Warning: student_config がないため、モデルレジストリに登録できません。")
            return {"error": "Student config was missing.", "metrics": final_metrics}

    async def prepare_dataset(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None
    ) -> Tuple[DataLoader, DataLoader]:
        
        # Collate関数の準備
        collate_fn_to_use: Callable[[List[Any]], Any]
        if collate_fn is None:
            collate_fn_factory = cast(TextCollateFnDef, collate_fn_orig_factory)
            collate_fn_to_use = collate_fn_factory(self.tokenizer, False)
        else:
            collate_fn_to_use = collate_fn

        # 教師モデルのロード
        teacher_model_instance = await self._get_or_load_teacher_model()

        # 修正: _PrecomputedDistillationDataset を使用してロジットを事前計算
        
        print("Pre-computing teacher logits for training data...")
        distill_train_dataset: Dataset = _PrecomputedDistillationDataset(
            original_dataset=train_dataset,
            teacher_model=teacher_model_instance,
            tokenizer=self.tokenizer,
            collate_fn_orig=collate_fn_to_use, 
            device=self.device,
            batch_size=batch_size
        )
        
        distill_val_dataset: Dataset
        if val_dataset:
            print("Pre-computing teacher logits for validation data...")
            distill_val_dataset = _PrecomputedDistillationDataset(
                original_dataset=val_dataset,
                teacher_model=teacher_model_instance,
                tokenizer=self.tokenizer,
                collate_fn_orig=collate_fn_to_use,
                device=self.device,
                batch_size=batch_size
            )
        else:
            # _PrecomputedDistillationDataset を split して使用
            try:
                train_len = len(cast(Sized, distill_train_dataset))
                train_size = int(0.9 * train_len)
                val_size = train_len - train_size
                if val_size == 0 and train_size > 0:
                     train_size -= 1
                     val_size = 1
                
                if train_size > 0 and val_size > 0:
                    distill_train_dataset, distill_val_dataset = torch.utils.data.random_split(distill_train_dataset, [train_size, val_size])
                else:
                    distill_val_dataset = distill_train_dataset
            except Exception as e:
                 logger.warning(f"Validation split failed: {e}. Using training set for validation.")
                 distill_val_dataset = distill_train_dataset

        # 蒸留用 Collate 関数 (ロジットとデータを結合する)
        distillation_collate_fn = self._create_distillation_collate_fn(
            collate_fn_orig=collate_fn_to_use 
        )

        # DataLoaderの作成
        # ロジットは既に計算済み（CPU Tensor）なので、num_workers を使っても安全
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

    def _create_distillation_collate_fn(
        self,
        collate_fn_orig: Callable[[List[Any]], Any]
    ) -> Callable:
        """
        通常のcollate_fnに加えて、事前計算された教師ロジットをパディングしてバッチ化するラッパー。
        """
        def distillation_collate(batch: List[Tuple[Dict[str, Any], torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # batch は (original_item, cached_logit) のタプルのリスト
            original_batch_items: List[Dict[str, Any]] = [item[0] for item in batch]
            teacher_logits_list: List[torch.Tensor] = [item[1] for item in batch]

            # 元のデータ（input_ids, targetsなど）をバッチ化
            collated_batch: Dict[str, torch.Tensor] = collate_fn_orig(original_batch_items)
            
            student_input: torch.Tensor
            attention_mask: torch.Tensor
            student_target: torch.Tensor
            
            if 'input_ids' in collated_batch:
                student_input = collated_batch['input_ids']
                attention_mask = collated_batch['attention_mask']
                student_target = collated_batch['labels']
            
            elif 'input_images' in collated_batch:
                student_input = collated_batch['input_images'] 
                student_target = collated_batch['labels']      
                attention_mask = torch.ones_like(student_target, dtype=torch.long) 
            
            else:
                raise KeyError(f"Neither 'input_ids' nor 'input_images' found in collated batch.")

            # 教師ロジットをパディングしてバッチ化
            # (Batch, SeqLen, VocabSize)
            padded_teacher_logits = torch.nn.utils.rnn.pad_sequence(
                teacher_logits_list, batch_first=True, padding_value=0.0
            )

            # 画像タスクの場合はシーケンス次元がないかもしれないので調整
            if student_input.dim() > 2: # Image task (B, C, H, W)
                # CNN出力が (B, 1, NumClasses) の場合などに備えてsqueeze
                if padded_teacher_logits.dim() == 3 and padded_teacher_logits.shape[1] == 1:
                    padded_teacher_logits = padded_teacher_logits.squeeze(1)
                return student_input, attention_mask, student_target, padded_teacher_logits

            # テキストタスクの場合、生徒と教師のシーケンス長が異なる場合の調整
            max_len_student = student_input.shape[1]
            max_len_teacher = padded_teacher_logits.shape[1]
            
            if max_len_student > max_len_teacher:
                # 教師側が短い -> 0埋め
                pad_size = max_len_student - max_len_teacher
                padding = torch.zeros(
                    (padded_teacher_logits.shape[0], pad_size, padded_teacher_logits.shape[2]),
                    dtype=padded_teacher_logits.dtype, device=padded_teacher_logits.device
                )
                padded_teacher_logits = torch.cat([padded_teacher_logits, padding], dim=1)
            
            elif max_len_teacher > max_len_student:
                # 生徒側が短い -> パディング
                pad_size = max_len_teacher - max_len_student
                pad_val_input = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                pad_val_target = -100
                
                padding_input = torch.full(
                    (student_input.shape[0], pad_size), pad_val_input,
                    dtype=student_input.dtype, device=student_input.device
                )
                student_input = torch.cat([student_input, padding_input], dim=1)

                padding_mask = torch.zeros(
                    (attention_mask.shape[0], pad_size),
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, padding_mask], dim=1)
                
                padding_target = torch.full(
                    (student_target.shape[0], pad_size), pad_val_target,
                    dtype=student_target.dtype, device=student_target.device
                )
                student_target = torch.cat([student_target, padding_target], dim=1)
            
            return student_input, attention_mask, student_target, padded_teacher_logits

        return distillation_collate
