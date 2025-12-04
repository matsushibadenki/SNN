# ファイルパス: snn_research/distillation/knowledge_distillation_manager.py
# (v9 修正版)
#
# Title: 知識蒸留 (Knowledge Distillation) 管理マネージャー
# Description:
# - ANN（教師モデル）からSNN（生徒モデル）への知識蒸留プロセス全体を管理・実行する。
# - タスク記述に基づき、モデルレジストリから教師/生徒モデルを取得・登録する。
# - データセットを知識蒸留形式（教師ロジットを含む）にラップする。
# - 蒸留トレーナー（DistillationTrainer）を呼び出して学習を実行する。
#
# 修正 (v9):
# - 循環インポートエラーを解消するため、collate_fn のインポート元を
#   `train.py` から `app/utils.py` に変更。
# - (v9 以前のmypyエラー修正コメントは省略)
#
# 修正 (v10): mypy エラー [name-defined], [assignment], [arg-type], [misc], [no-redef], [list-item] を修正
# 修正 (v11): mypy エラー [syntax] (インデント) を修正
#
# 修正 (v_async_fix):
# - L333: prepare_dataset を async def に変更。
# - L345: asyncio.run() を await に変更。
#
# 修正 (v_hpo_fix_callable_error):
# - DIコンテナから渡された config は解決済みの値 (dict) を OmegaConf に
#   変換したものであるため、.log_dir() のような関数呼び出しを
#   .log_dir のような属性アクセスに修正。
#
# 修正 (v_hpo_fix_key_error):
# - _DistillationWrapperDataset (L575) と distillation_collate (L442) が 
#   'input_ids' だけでなく 'input_images' も処理できるように修正。
#
# 修正 (v_refactor_fix):
# - [name-defined] エラーを修正するため、torchvision.models をインポート。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
# --- ▼ 修正: 必要な型ヒントをインポート ▼ ---
from typing import Dict, Any, Optional, List, Callable, Tuple, cast, TypeAlias, Sized
import os
import json
import logging
import asyncio # [name-defined] asyncio をインポート
# --- ▲ 修正 ▲ ---
from omegaconf import DictConfig

# --- ▼ 修正: [name-defined] エラー修正 ▼ ---
import torchvision.models as models  # type: ignore[import-untyped]
# --- ▲ 修正 ▲ ---

from snn_research.distillation.model_registry import ModelRegistry
# --- ▼ 修正: [name-defined] DistillationTrainer をインポート ▼ ---
from snn_research.training.trainers import DistillationTrainer
# --- ▲ 修正 ▲ ---
from snn_research.benchmark.metrics import calculate_accuracy
# ◾️◾️◾️ 修正: [name-defined] mypyエラー回避のため、型ヒントをインポート ◾️◾️◾️
from torch.optim.lr_scheduler import LRScheduler
# ◾️◾️◾️ 修正終わり ◾️◾️◾️

logger = logging.getLogger(__name__)

# --- ▼ 修正: 型エイリアスを TypeAlias を使ってファイル先頭で定義 ▼ ---
TextCollateFnDef: TypeAlias = Callable[[PreTrainedTokenizerBase, bool], Callable[[List[Any]], Any]]
# --- ▲ 修正 ▲ ---

# --- ▼▼▼ 修正 (v9): インポート元を train.py から app.utils.py に変更 ▼▼▼ ---
try:
    # collate_fn は app/utils.py に定義されている
    from app.utils import collate_fn as text_collate_fn
    
    collate_fn_orig_factory: TextCollateFnDef = cast(TextCollateFnDef, text_collate_fn)
    logger.info("Successfully imported collate_fn from app.utils.py.")
except ImportError:
    logger.warning("Warning: Could not import collate_fn from app.utils.py. Using fallback definition.")
    # フォールバック (主に型チェックのため)
    def _fallback_collate(batch: List[Any]) -> Any:
        raise NotImplementedError("Fallback collate_fn called. Check app/utils.py.")
    
    def fallback_collate_fn_def(tokenizer: PreTrainedTokenizerBase, is_distillation: bool) -> Callable[[List[Any]], Any]:
        return _fallback_collate
    
    # --- ▼ 修正: [no-redef] [misc] [list-item] エラー解消のため、重複定義を削除 ▼ ---
    # (TextCollateFnDef は 43行目で定義済み)
    collate_fn_orig_factory = fallback_collate_fn_def
    # --- ▲ 修正 ▲ ---
# --- ▲▲▲ 修正 (v9) ▲▲▲ ---


class KnowledgeDistillationManager:
    """
    SNNへの知識蒸留プロセス全体をオーケストレーションする。
    """
    def __init__(
        self,
        student_model: nn.Module,
        trainer: DistillationTrainer, # <-- [name-defined] 修正
        model_registry: ModelRegistry,
        device: str,
        config: DictConfig, # ◾️ config を追加
        teacher_model: Optional[nn.Module] = None,
        teacher_model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_model_name = teacher_model_name
        self.tokenizer_name = tokenizer_name
        self.trainer = trainer
        self.model_registry = model_registry
        self.device = device
        # ◾️◾️◾️ 修正: config をインスタンス変数として保持 ◾️◾️◾️
        self.config = config 
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️

        if not teacher_model and not teacher_model_name:
            raise ValueError("Either teacher_model (instance) or teacher_model_name (str) must be provided.")
            
        if not tokenizer_name and not (isinstance(teacher_model_name, str) and teacher_model_name):
             raise ValueError("tokenizer_name or a valid teacher_model_name must be provided to load tokenizer.")

        self.tokenizer_name = tokenizer_name if tokenizer_name else cast(str, teacher_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # ◾️◾️◾️ 修正(mypy v8): energy.py への移管に伴い削除 ◾️◾️◾️
        # self.energy_metrics = EnergyMetrics(...)
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️

    async def _get_or_load_teacher_model(self) -> nn.Module:
        """
        教師モデルのインスタンスを取得する。
        インスタンスが提供されていればそれを返し、なければ名前からロードする。
        """
        if self.teacher_model:
            return self.teacher_model.to(self.device).eval()

        if not self.teacher_model_name:
             raise ValueError("Cannot load teacher model: teacher_model_name is not set.")

        print(f"🧠 Loading teacher model '{self.teacher_model_name}' from Hugging Face...")
        try:
            # --- ▼ 修正 (v_hpo_fix_key_error): 画像タスク (resnet) のロードに対応 ▼ ---
            if self.teacher_model_name == "resnet18":
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                num_ftrs = model.fc.in_features
                model.fc = torch.nn.Linear(num_ftrs, 10) # CIFAR-10前提
            else:
                model = AutoModelForCausalLM.from_pretrained(self.teacher_model_name)
            # --- ▲ 修正 (v_hpo_fix_key_error) ▲ ---
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
        """
        タスク記述に基づき、オンデマンドで専門家モデルを学習させるパイプライン。
        """
        print(f"--- On-Demand Learning Pipeline Initiated ---")
        print(f"Task: {task_description}")

        # 1. 既存の専門家モデルを検索
        if not force_retrain:
            existing_experts = await self.model_registry.find_models_for_task(task_description, top_k=1)
            if existing_experts:
                best_expert = existing_experts[0]
                # ◾️◾️◾️ 修正: mypyエラー [assignment] を修正 ◾️◾️◾️
                best_expert['model_id'] = task_description # type: ignore[assignment]
                # ◾️◾️◾️ 修正終わり ◾️◾️◾️
                print(f"✅ Found existing expert: {best_expert.get('model_path')}")
                return best_expert

        print(f"ℹ️ No suitable expert found or retraining forced. Starting new training.")

        # 2. 学習データの準備 (Web Crawlerが生成した .jsonl を想定)
        if not os.path.exists(unlabeled_data_path):
            print(f"❌ Error: Unlabeled data file not found at '{unlabeled_data_path}'")
            return {"error": "Data file not found"}
        
        # ◾️◾️◾️ 修正: mypyエラー [assignment] を修正 ◾️◾️◾️
        from snn_research.data.datasets import SimpleTextDataset # 循環インポートを避けるため局所インポート
        # ◾️◾️◾️ 修正終わり ◾️◾️◾️
        
        try:
            # --- ▼ 修正 (v_hpo_fix_callable_error): .time_steps() -> .time_steps ▼ ---
            train_dataset_raw = SimpleTextDataset(
                file_path=unlabeled_data_path,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.model.time_steps # type: ignore[attr-defined] 
            )
            # --- ▲ 修正 (v_hpo_fix_callable_error) ▲ ---
            
            # データセットが小さすぎる場合のフォールバック
            if len(cast(Sized, train_dataset_raw)) < 10:
                 print(f"⚠️ Warning: Dataset at '{unlabeled_data_path}' is too small ({len(cast(Sized, train_dataset_raw))} samples).")
                 if len(cast(Sized, train_dataset_raw)) == 0:
                     return {"error": "No data found in the provided file."}
                 # データを複製して最小限のバッチ数を確保
                 train_dataset_raw = torch.utils.data.ConcatDataset([train_dataset_raw] * (10 // len(cast(Sized, train_dataset_raw)) + 1)) # type: ignore[assignment]


            # 蒸留用にデータセットをラップし、教師モデルのロジットを事前計算
            print("Preparing distillation dataset (pre-calculating teacher logits)...")
            
            # --- ▼ 修正 (v_hpo_fix_callable_error): .batch_size() -> .batch_size ▼ ---
            train_loader, val_loader = await self.prepare_dataset( # type: ignore[call-arg]
                train_dataset_raw,
                None, # 検証セットはここでは作成しない (簡易化のため)
                batch_size=self.config.training.batch_size, # type: ignore[attr-defined]
                collate_fn=None # prepare_dataset内部でcollate_fnが生成される
            )
            # --- ▲ 修正 (v_hpo_fix_callable_error) ▲ ---

        except Exception as e:
            print(f"❌ Error preparing dataset: {e}")
            return {"error": f"Dataset preparation failed: {e}"}

        # 3. 蒸留の実行
        # --- ▼ 修正 (v_hpo_fix_callable_error): .epochs() -> .epochs ▼ ---
        print(f"Starting distillation training for {self.config.training.epochs} epochs...") # type: ignore[attr-defined]
        
        final_metrics: Dict[str, Any] = await self.run_distillation( # type: ignore[assignment]
            train_loader=train_loader,
            val_loader=val_loader, # 検証セット
            epochs=self.config.training.epochs, # type: ignore[attr-defined]
            model_id=task_description, # タスク記述をモデルIDとして使用
            task_description=task_description,
            student_config=student_config # 渡されたSNNモデル設定
        )
        # --- ▲ 修正 (v_hpo_fix_callable_error) ▲ ---

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
        """
        知識蒸留の学習と評価のループを実行する。
        """
        best_metric = float('inf') # 損失を最小化
        best_model_path = ""
        
        # --- ▼ 修正 (v_hpo_fix_callable_error): .log_dir() -> .log_dir ▼ ---
        log_dir = self.config.training.log_dir # type: ignore[attr-defined]
        # --- ▲ 修正 (v_hpo_fix_callable_error) ▲ ---
        os.makedirs(log_dir, exist_ok=True)

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            # --- 訓練 ---
            train_metrics = self.trainer.train_epoch(train_loader, epoch)
            
            # --- 検証 ---
            if val_loader:
                val_metrics = self.trainer.evaluate(val_loader, epoch)
                
                # メトリクス名 (loss or accuracy)
                metric_name = self.config.training.get("metric_to_optimize", "total") # type: ignore[attr-defined]
                current_metric = val_metrics.get(metric_name, float('inf'))

                print(f"Epoch {epoch + 1} Validation Metrics: {val_metrics}")

                # ベストモデルの保存
                if current_metric < best_metric:
                    best_metric = current_metric
                    best_model_path = os.path.join(log_dir, f"{model_id}_best.pth")
                    
                    config_to_save: Dict[str, Any] = student_config if student_config is not None else {} # type: ignore[assignment]
                    
                    self.trainer.save_checkpoint(
                        path=best_model_path,
                        epoch=epoch,
                        metric_value=best_metric,
                        config=config_to_save, # ◾️ モデル設定を保存
                        tokenizer_name=self.tokenizer_name
                    )
            else:
                 # 検証ローダーがない場合は、訓練メトリクスで代用（非推奨）
                 best_metric = train_metrics.get("total", float('inf'))


        # --- 最終評価とモデル登録 ---
        print("\n--- Final Evaluation on Validation Set ---")
        final_metrics: Dict[str, Any] = {"accuracy": 0.0, "avg_spikes_per_sample": float('inf')}
        
        if val_loader:
            # 最高のチェックポイントをロード
            if os.path.exists(best_model_path):
                self.trainer.load_checkpoint(best_model_path)
            
            final_eval_metrics_raw = self.trainer.evaluate(val_loader, epochs)
            
            final_metrics['accuracy'] = final_eval_metrics_raw.get('accuracy', 0.0) # type: ignore[assignment]
            final_metrics['avg_spikes_per_sample'] = final_eval_metrics_raw.get('avg_cutoff_steps', 0.0) # type: ignore[assignment]
            
        print(f"Final Metrics: {final_metrics}")

        # モデルレジストリに登録
        if student_config:
            await self.model_registry.register_model(
                model_id=model_id,
                task_description=task_description,
                metrics=final_metrics,
                model_path=best_model_path,
                config=student_config
            )
            
            # 登録した情報を返す
            final_model_info: Dict[str, Any] = { # type: ignore[assignment]
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

    # --- ▼ 修正 (v_async_fix): async def に変更 ▼ ---
    async def prepare_dataset(
    # --- ▲ 修正 (v_async_fix) ▲ ---
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        教師モデルのロジットを事前計算するデータセットラッパーを適用する。
        """
        
        # --- ▼ 修正 (v_hpo_fix_key_error): collate_fn の処理ロジックを修正 ▼ ---
        collate_fn_to_use: Callable[[List[Any]], Any]
        
        if collate_fn is None:
            # ケース1: (textタスク) collate_fnが指定されなかった場合、デフォルトのテキスト用ファクトリを使用
            collate_fn_factory = cast(TextCollateFnDef, text_collate_fn)
            collate_fn_to_use = collate_fn_factory(self.tokenizer, False)
        else:
            # ケース2: (cifar10タスク) task.get_collate_fn() が渡された場合
            collate_fn_to_use = collate_fn
        # --- ▲ 修正 (v_hpo_fix_key_error) ▲ ---

        # --- ▼ 修正 (v_async_fix): asyncio.run() を await に変更 ▼ ---
        teacher_model_instance = await self._get_or_load_teacher_model()
        # --- ▲ 修正 (v_async_fix) ▲ ---

        # 蒸留用データセットラッパー
        distill_train_dataset: Dataset = _DistillationWrapperDataset(
            original_dataset=train_dataset,
            teacher_model=teacher_model_instance,
            tokenizer=self.tokenizer,
            # --- ▼ 修正 (v_hpo_fix_key_error): ファクトリではなく、インスタンスを渡す ▼ ---
            collate_fn_orig=collate_fn_to_use, 
            # --- ▲ 修正 (v_hpo_fix_key_error) ▲ ---
            device=self.device
        )
        
        distill_val_dataset: Dataset
        if val_dataset:
            distill_val_dataset = _DistillationWrapperDataset(
                original_dataset=val_dataset,
                teacher_model=teacher_model_instance,
                tokenizer=self.tokenizer,
                # --- ▼ 修正 (v_hpo_fix_key_error): ファクトリではなく、インスタンスを渡す ▼ ---
                collate_fn_orig=collate_fn_to_use,
                # --- ▲ 修正 (v_hpo_fix_key_error) ▲ ---
                device=self.device
            )
        else:
            # 検証セットがない場合、訓練セットから10%を拝借 (簡易的)
            try:
                train_size = int(0.9 * len(cast(Sized, distill_train_dataset)))
                val_size = len(cast(Sized, distill_train_dataset)) - train_size
                if val_size == 0 and train_size > 0:
                     train_size -= 1
                     val_size = 1
                
                if train_size > 0 and val_size > 0:
                    distill_train_dataset, distill_val_dataset = torch.utils.data.random_split(distill_train_dataset, [train_size, val_size])
                else:
                    print("Warning: Dataset too small to split for validation. Using training set for validation.")
                    distill_val_dataset = distill_train_dataset
            except Exception as e:
                 print(f"Warning: Could not split dataset for validation: {e}. Using training set for validation.")
                 distill_val_dataset = distill_train_dataset


        # 蒸留用の collate_fn (タプルを返す)
        distillation_collate_fn = self._create_distillation_collate_fn(
            # --- ▼ 修正 (v_hpo_fix_key_error): ファクトリではなく、インスタンスを渡す ▼ ---
            collate_fn_orig=collate_fn_to_use 
            # --- ▲ 修正 (v_hpo_fix_key_error) ▲ ---
        )

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
        # --- ▼ 修正 (v_hpo_fix_key_error): シグネチャを修正 ▼ ---
        collate_fn_orig: Callable[[List[Any]], Any]
        # --- ▲ 修正 (v_hpo_fix_key_error) ▲ ---
    ) -> Callable:
        """
        知識蒸留用のデータローダー collate_fn を作成する。
        (student_input, attention_mask, student_target, teacher_logits) のタプルを返す。
        """
        
        def distillation_collate(batch: List[Tuple[Dict[str, Any], torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Args:
                batch (List[Tuple[Dict, Tensor]]): 
                    _DistillationWrapperDataset からの出力。
                    各要素は (original_batch_item, teacher_logits_for_item) のタプル。
            """
            
            original_batch_items: List[Dict[str, Any]] = [item[0] for item in batch]
            teacher_logits_list: List[torch.Tensor] = [item[1] for item in batch]

            # 1. 元の collate_fn を使って、テキストデータをテンソル化 (SNN入力用)
            collated_batch: Dict[str, torch.Tensor] = collate_fn_orig(original_batch_items)
            
            # --- ▼ 修正 (v_hpo_fix_key_error): 'input_ids' と 'input_images' の両方に対応 ▼ ---
            # (student_input, attention_mask, student_target) を決定する
            
            student_input: torch.Tensor
            attention_mask: torch.Tensor
            student_target: torch.Tensor
            
            if 'input_ids' in collated_batch:
                # Text Task
                student_input = collated_batch['input_ids']
                attention_mask = collated_batch['attention_mask']
                student_target = collated_batch['labels']
            
            elif 'input_images' in collated_batch:
                # Image Task
                student_input = collated_batch['input_images'] # (B, C, H, W)
                student_target = collated_batch['labels']      # (B,)
                # 画像タスクにはシーケンスの attention_mask はない
                attention_mask = torch.ones_like(student_target, dtype=torch.long) # (B,)
            
            else:
                raise KeyError(f"Neither 'input_ids' nor 'input_images' found in collated batch from original collate_fn. Keys: {collated_batch.keys()}")
            # --- ▲ 修正 (v_hpo_fix_key_error) ▲ ---


            # 2. 教師ロジットをパディングしてバッチ化
            #    teacher_logits_list の各要素は (SeqLen_item, VocabSize)
            padded_teacher_logits = torch.nn.utils.rnn.pad_sequence(
                teacher_logits_list, batch_first=True, padding_value=0.0
            )

            # --- ▼ 修正 (v_hpo_fix_key_error): 画像タスクの形状 (B, 1, C) に対応 ▼ ---
            # 3. シーケンス長の整合性を取る
            
            # 画像タスクの場合、student_input は (B, C, H, W)、
            # padded_teacher_logits は (B, 1, NumClasses)
            # DistillationLoss (losses.py) は is_classification (ndim==2) をチェックする
            # ここで形状を合わせる必要はない。
            
            if student_input.dim() > 2: # Image task (B, C, H, W)
                # student_target は (B,)
                # padded_teacher_logits は (B, 1, NumClasses) -> (B, NumClasses) にする
                padded_teacher_logits = padded_teacher_logits.squeeze(1)
                
                # (student_input, attention_mask(dummy), student_target, teacher_logits)
                return student_input, attention_mask, student_target, padded_teacher_logits

            # --- Text Task の場合の整合性チェック ---
            max_len_student = student_input.shape[1]
            max_len_teacher = padded_teacher_logits.shape[1]
            
            if student_target.shape[1] != max_len_student:
                 pass

            if max_len_student > max_len_teacher:
                pad_size = max_len_student - max_len_teacher
                padding = torch.zeros(
                    (padded_teacher_logits.shape[0], pad_size, padded_teacher_logits.shape[2]),
                    dtype=padded_teacher_logits.dtype, device=padded_teacher_logits.device
                )
                padded_teacher_logits = torch.cat([padded_teacher_logits, padding], dim=1)
            
            elif max_len_teacher > max_len_student:
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
            # --- ▲ 修正 (v_hpo_fix_key_error) ▲ ---

        return distillation_collate


class _DistillationWrapperDataset(Dataset):
    """
    既存のデータセットをラップし、教師モデルの推論を事前実行して
    (item, teacher_logits) のペアを返すデータセット。
    """
    def __init__(
        self,
        original_dataset: Dataset,
        teacher_model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        # --- ▼ 修正 (v_hpo_fix_key_error): シグネチャを修正 ▼ ---
        collate_fn_orig: Callable[[List[Any]], Any],
        # --- ▲ 修正 (v_hpo_fix_key_error) ▲ ---
        device: str
    ):
        self.original_dataset = original_dataset
        self.teacher_model = teacher_model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        
        # --- ▼ 修正 (v_hpo_fix_key_error): ファクトリ呼び出しを削除 ▼ ---
        self.collate_fn_orig = collate_fn_orig
        # --- ▲ 修正 (v_hpo_fix_key_error) ▲ ---
        
        if self.collate_fn_orig is None:
             raise RuntimeError("collate_fn_orig is None, cannot process item.")
        
        logger.info(f"DistillationWrapperDataset initialized for {len(cast(Sized, self.original_dataset))} samples.")

    def __len__(self) -> int:
        return len(cast(Sized, self.original_dataset))

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        """
        元のアイテムと、それに対する教師モデルのロジットを返す。
        """
        # 1. 元のデータセットからアイテムを取得
        original_item: Any = self.original_dataset[idx]
        
        # 2. collate_fn を使って、単一アイテムをバッチ形式のテンソルに変換
        if self.collate_fn_orig is None:
             raise RuntimeError("collate_fn_orig is None, cannot process item.")
        
        collated_batch: Dict[str, torch.Tensor] = self.collate_fn_orig([original_item])
        
        # 3. 教師モデルでロジットを計算
        
        # --- ▼ 修正 (v_hpo_fix_key_error): 'input_ids' と 'input_images' の両方に対応 ▼ ---
        teacher_logits: torch.Tensor
        
        if 'input_ids' in collated_batch:
            # --- Text Task ---
            input_ids = collated_batch['input_ids'].to(self.device)
            attention_mask = collated_batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits # (B=1, SeqLen, VocabSize)
        
        elif 'input_images' in collated_batch:
            # --- Image Task (e.g., CIFAR-10) ---
            input_images = collated_batch['input_images'].to(self.device)
            # ResNetモデルはキーワード引数なしでテンソルを受け取る
            teacher_logits = self.teacher_model(input_images) # (B=1, NumClasses)
            
            # (B, C) -> (B, 1, C) に拡張し、テキストタスク (B, S, V) と形状を合わせる
            if teacher_logits.dim() == 2:
                teacher_logits = teacher_logits.unsqueeze(1) # (B=1, 1, NumClasses)
        
        else:
            raise KeyError(f"Neither 'input_ids' nor 'input_images' found in collated batch. Keys: {collated_batch.keys()}")
        # --- ▲ 修正 (v_hpo_fix_key_error) ▲ ---

        # 4. CPUに移動し、バッチ次元を削除
        teacher_logits_cpu = teacher_logits.squeeze(0).cpu().to(torch.float16) # (SeqLen or 1, VocabSize or NumClasses)
        
        # (元のアイテム, 教師ロジット) のタプルを返す
        return original_item, teacher_logits_cpu