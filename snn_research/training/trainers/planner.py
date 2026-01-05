# ファイルパス: snn_research/training/trainers/planner.py
# 日本語タイトル: Planner Trainer v2.3 - Mypy Fix
# 目的・内容:
#   HierarchicalPlannerおよびPlannerSNNの学習を担当するトレーナークラス。
#   - mypyエラー修正: self.optimizer が None でないことを明示的にチェック(assert)する処理を追加。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
import logging

from snn_research.training.base_trainer import AbstractTrainer
from snn_research.cognitive_architecture.planner_snn import PlannerSNN

logger = logging.getLogger(__name__)


class PlannerTrainer(AbstractTrainer):
    """
    PlannerSNNモデルの学習を行うトレーナー。
    AbstractTrainerを継承し、train_epoch / validate を実装する。
    """

    def __init__(
        self,
        model: PlannerSNN,
        optimizer: torch.optim.Optimizer,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[Dict[str, Any]] = None
    ):
        # 親クラスの初期化
        # 親クラスの初期化 (args: model, config, optimizer, ...)
        super().__init__(model, config=config, optimizer=optimizer, device=device)
        self.model = model  # 型ヒントのために再代入(実体は親クラスと同じ)
        self.tokenizer = tokenizer
        # self.config is already set by super().__init__ as DictConfig
        self.criterion = nn.CrossEntropyLoss()

        logger.info(f"🗺️ PlannerTrainer initialized on {device}.")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        1エポック分の学習ループ。
        """
        # mypy fix: OptimizerがNoneでないことを保証
        assert self.optimizer is not None, "Optimizer is required for training."

        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        for batch in train_loader:
            self.optimizer.zero_grad()

            # 1. データの展開と前処理
            goal_texts = batch["goal_text"]
            target_skill_ids = batch["skill_id"].to(self.device)

            # トークナイズ
            encoded_inputs = self.tokenizer(
                goal_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            ).input_ids.to(self.device)

            # 2. Forward Pass
            logits = self.model(encoded_inputs)

            # 3. Loss Calculation
            loss = self.criterion(logits, target_skill_ids)

            # 4. Backward Pass & Optimize
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 5. Metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == target_skill_ids).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            steps += 1

            self.global_step += 1

        if steps == 0:
            return {"train_loss": 0.0, "train_accuracy": 0.0}

        return {
            "train_loss": total_loss / steps,
            "train_accuracy": total_acc / steps
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        検証ループ。
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        with torch.no_grad():
            for batch in val_loader:
                goal_texts = batch["goal_text"]
                target_skill_ids = batch["skill_id"].to(self.device)

                encoded_inputs = self.tokenizer(
                    goal_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=128
                ).input_ids.to(self.device)

                logits = self.model(encoded_inputs)
                loss = self.criterion(logits, target_skill_ids)

                preds = torch.argmax(logits, dim=1)
                acc = (preds == target_skill_ids).float().mean().item()

                total_loss += loss.item()
                total_acc += acc
                steps += 1

        if steps == 0:
            return {"val_loss": 0.0, "val_accuracy": 0.0}

        return {
            "val_loss": total_loss / steps,
            "val_accuracy": total_acc / steps
        }
