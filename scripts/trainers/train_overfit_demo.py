# ファイルパス: scripts/trainers/train_overfit_demo.py
# 日本語タイトル: BitSpikeMamba 過学習デモ (バグ修正 & 高速検証版)
# 目的: 1.58bitモデルの基本動作を検証する。ヘルスチェック用にステップ制限に対応。

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
import argparse

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyTextDataset(Dataset):
    """検証用のダミーテキストデータセット。"""
    def __init__(self, vocab_size=1000, seq_len=16, num_samples=32):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y

def train_demo(epochs=1, max_steps=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Starting BitSpikeMamba training demo on {device}")

    # 1. モデルとデータの準備
    vocab_size = 1000
    model = BitSpikeMamba(
        d_model=128, d_state=16, d_conv=4, expand=2, 
        num_layers=2, time_steps=4, 
        neuron_config={"threshold": 1.0}, vocab_size=vocab_size
    ).to(device)

    dataset = DummyTextDataset(vocab_size=vocab_size)
    # [Fix] ここで dataloader を正しく定義
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 2. 学習ループ
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        # 明示的に定義された dataloader を使用
        for i, (x, y) in enumerate(dataloader):
            if max_steps is not None and i >= max_steps:
                break
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 順伝播
            logits, _, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
            # 逆伝播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 2 == 0:
                logger.info(f"  Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

    logger.info("✅ Training demo finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--max_steps", type=int, default=5, help="Max steps per epoch for quick health-check")
    args = parser.parse_args()
    
    train_demo(epochs=args.epochs, max_steps=args.max_steps)
