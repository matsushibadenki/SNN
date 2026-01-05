# ファイルパス: scripts/runners/train_planner.py
# 日本語タイトル: Planner Training Runner v2.2 - Tokenizer Fix
# 目的・内容:
#   PlannerSNNモデルを訓練するスクリプト。
#   修正: トークナイザーのpad_token設定を追加し、ValueErrorを回避。

from snn_research.training.trainers.planner import PlannerTrainer
from app.containers import TrainingContainer
import sys
import os
import argparse
import torch
import json
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict, Any

# プロジェクトルートの設定
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# --- Custom Dataset Definition ---

class PlannerDataset(Dataset):
    """PlannerTrainerが期待する形式 {goal_text, skill_id} を返すデータセット"""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, Any]:
        text = self.texts[idx]
        return {
            "goal_text": text,
            "skill_id": torch.tensor(self._get_skill_id(text), dtype=torch.long)
        }

    def _get_skill_id(self, text: str) -> int:
        # 簡易的なルールベースのラベリング
        lower_text = text.lower()
        if any(kw in lower_text for kw in ["summarize", "what is", "explain", "tell me"]):
            return 0  # Knowledge / QA Skill
        if any(kw in lower_text for kw in ["sentiment", "feel", "enjoy", "hate", "love"]):
            return 1  # Empathy / Emotional Skill
        if any(kw in lower_text for kw in ["plan", "organize", "schedule", "create"]):
            return 2  # Planning Skill
        return 0  # Default


def main():
    parser = argparse.ArgumentParser(description="SNN Planner Trainer")
    parser.add_argument(
        "--config", type=str, default="configs/templates/base_config.yaml", help="Base config file path")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml",
                        help="Model architecture config file path")
    parser.add_argument("--data_path", type=str,
                        default="data/sample_data.jsonl", help="Path to the training data.")
    args = parser.parse_args()

    # 1. コンテナの初期化
    container = TrainingContainer()
    container.config.from_yaml(args.config)
    container.config.from_yaml(args.model_config)

    device = container.device()
    print(f"🚀 Training PlannerSNN on {device}")

    # 2. コンポーネントの取得
    planner_model = container.planner_snn()
    planner_optimizer = container.planner_optimizer(
        params=planner_model.parameters())

    # ★ 修正: トークナイザーのパディングトークン設定 ★
    tokenizer = container.tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("ℹ️ Set tokenizer.pad_token to tokenizer.eos_token")

    # 学習率スケジューラ
    epochs = container.config.training.epochs()
    if not isinstance(epochs, int):
        epochs = 10

    scheduler = CosineAnnealingLR(planner_optimizer, T_max=epochs)

    # 3. Trainerの初期化
    trainer = PlannerTrainer(
        model=planner_model,
        optimizer=planner_optimizer,
        tokenizer=tokenizer,
        device=device,
        config=container.config()
    )

    # 4. データセットの準備
    if not os.path.exists(args.data_path):
        print(
            f"⚠️ Data file not found at {args.data_path}. Generating dummy data...")
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        dummy_data = [
            {"text": "Please summarize this article."},
            {"text": "How do you feel about the weather?"},
            {"text": "Make a plan for the trip."},
            {"text": "Explain quantum physics."},
            {"text": "I love this song!"},
            {"text": "Organize my schedule."},
            {"text": "Create a todo list for today."}
        ]
        with open(args.data_path, 'w') as f:
            for item in dummy_data:
                f.write(json.dumps(item) + "\n")

    texts = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    texts.append(json.loads(line)['text'])
                except json.JSONDecodeError:
                    pass

    if not texts:
        print("❌ No valid data found. Exiting.")
        return

    dataset = PlannerDataset(texts)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True)  # Batch size小さめに

    # 5. 訓練ループの実行
    print(">>> Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        trainer.current_epoch = epoch

        metrics = trainer.train_epoch(dataloader)
        scheduler.step()

        print(
            f"Epoch {epoch}/{epochs} | Loss: {metrics['train_loss']:.4f} | Acc: {metrics['train_accuracy']:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % 5 == 0 or epoch == epochs:
            trainer.save_checkpoint(
                f"planner_epoch_{epoch}.pth", metric=metrics['train_accuracy'])

    print("✅ Planner training finished successfully.")


if __name__ == "__main__":
    main()
