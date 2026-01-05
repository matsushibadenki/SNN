# ファイルパス: snn_research/training/trainers/probabilistic.py
# Title: Probabilistic Learning Trainer
# Description:
#   BPを使わず、局所学習則（Predictive Coding, STDP等）のみで学習を進めるトレーナー。
#   AbstractSNNNetwork の run_learning_step を使用する。
#   修正: AbstractTrainer への継承変更

import torch
import logging
from typing import Dict, Any

from snn_research.training.base_trainer import AbstractTrainer, DataLoader
from snn_research.core.networks.abstract_snn_network import AbstractSNNNetwork

logger = logging.getLogger(__name__)


class ProbabilisticTrainer(AbstractTrainer):
    """
    局所学習則に基づくトレーナー。
    勾配降下法(Backprop)は使用しないか、補助的にのみ使用する。
    """

    def __init__(self, model: AbstractSNNNetwork, **kwargs: Any):
        # AbstractTrainer の初期化
        super().__init__(model, **kwargs)
        self.model: AbstractSNNNetwork = model  # 型ヒントの絞り込み

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()

        update_magnitude = 0.0
        steps = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)

            # 1. 順伝播 (状態の更新)
            # SNNの場合は時間方向の展開が必要な場合があるが、
            # ここではネットワーク内部で処理されるか、静的入力として扱う
            self.model.reset_state()
            _ = self.model(data)

            # 2. 学習ステップ (局所学習則の適用)
            # targets は教師あり学習の場合に使用（PCなど）
            stats = self.model.run_learning_step(data, targets)

            # 統計情報の集約
            batch_updates = sum(v.item()
                                for k, v in stats.items() if 'magnitude' in k)
            update_magnitude += batch_updates

            # (オプション) 教師あり損失の記録
            # 実際の重み更新は run_learning_step 内で行われているため、ここでは監視のみ
            steps += 1
            self.global_step += 1

        avg_update = update_magnitude / max(1, steps)
        logger.info(
            f"Epoch {self.current_epoch}: Avg Weight Update Magnitude: {avg_update:.6f}")

        self.current_epoch += 1
        return {'update_magnitude': avg_update}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                self.model.reset_state()

                # 推論実行
                # PC Networkの場合、出力は最終層の状態（クラス予測等）
                outputs = self.model(data)

                if outputs is not None:
                    # 分類タスクを想定
                    if outputs.shape[1] == 10:  # CIFAR-10 etc.
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

        acc = 100 * correct / max(1, total)
        logger.info(f"Validation Accuracy: {acc:.2f}%")
        return {'accuracy': acc}
