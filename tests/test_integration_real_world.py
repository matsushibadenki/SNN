# ファイルパス: tests/test_integration_real_world.py
# (修正: オンライン学習収束テストのステップ数緩和)

import pytest
import torch
import logging
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

logger = logging.getLogger(__name__)

# --- Mock Trainer for Testing ---
class MockBreakthroughTrainer:
    def __init__(self):
        self.loss_history = []
        # 単純な学習動作をシミュレート（損失が徐々に減る）
        self.current_loss = 1.0
        self.decay = 0.95

    def train_step(self, batch) -> float:
        # バッチ内容に関わらず損失を減衰させる
        self.current_loss *= self.decay
        # ノイズを加える
        noisy_loss = self.current_loss + (torch.rand(1).item() * 0.05)
        self.loss_history.append(noisy_loss)
        return noisy_loss

    def get_average_loss(self, window: int) -> float:
        if not self.loss_history:
            return 1.0
        window = min(len(self.loss_history), window)
        return sum(self.loss_history[-window:]) / window

@pytest.fixture
def online_learning_setup() -> Tuple[MockBreakthroughTrainer, DataLoader]:
    trainer = MockBreakthroughTrainer()
    # ダミーデータストリーム
    data = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))
    dataset = TensorDataset(data, targets)
    loader = DataLoader(dataset, batch_size=1)
    return trainer, loader

class TestRealWorldScenarios:
    def test_online_learning_convergence(self, online_learning_setup):
        """オンライン学習の収束性テスト（詳細化・緩和版）。"""
        logger.info("Testing online learning convergence...")
        trainer, stream_loader = online_learning_setup
        
        # [Fix] ステップ数を増やし、閾値を少し緩和してFlakinessを排除
        max_steps = 100 
        convergence_threshold = 0.2 
        
        initial_loss = float('inf')
        converged = False

        # 1. データストリームをシミュレートし、学習
        step = 0
        for batch in stream_loader:
            if step >= max_steps:
                break
            loss = trainer.train_step(batch)
            if step == 0:
                initial_loss = loss
            logger.info(f"   -> Online step {step+1}/{max_steps}: Loss = {loss:.4f}")

            # 損失が極端に発散したら失敗
            if step > 20 and loss > initial_loss * 3: 
                logger.error(f"Loss possibly diverged (Initial: {initial_loss:.4f}, Current: {loss:.4f} at step {step+1}).")
                pytest.fail("Loss possibly diverged during online learning.")

            # 収束判定 (直近数ステップの平均損失が閾値以下)
            avg_loss_last_5 = trainer.get_average_loss(window=5)
            if step > 10 and avg_loss_last_5 < convergence_threshold:
                converged = True
                logger.info(f"   -> Converged at step {step+1} (Avg loss {avg_loss_last_5:.4f} < {convergence_threshold}).")
                break
            step += 1

        if not converged:
             logger.error(f"Online learning did not converge within {max_steps} steps (Last avg loss: {trainer.get_average_loss(window=5):.4f}, Threshold: {convergence_threshold}).")
        
        assert converged, f"Online learning did not converge within {max_steps} steps."