# ファイルパス: scripts/trainers/train_overfit_demo.py
# 日本語タイトル: Brain v20 Overfit Trainer
# 目的・内容:
#   デモ用に特定のテキストを短時間で「丸暗記」させるスクリプト。
#   Lossを1.0以下まで下げ、まともな文章生成能力を持たせることを最優先する。

import os
import sys
import logging
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("OverfitTrainer")

class ShakespeareDataset(Dataset):
    def __init__(self, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        # 有名なフレーズを繰り返して確実に覚えさせる
        text = (
            "To be, or not to be, that is the question: "
            "Whether 'tis nobler in the mind to suffer "
            "The slings and arrows of outrageous fortune, "
            "Or to take arms against a sea of troubles "
            "And by opposing end them. "
            "Good evening, my friend. I am Brain v20, a neuromorphic intelligence. "
            "I think, therefore I am. "
        ) * 100 # データ量を水増し
        
        self.data = tokenizer.encode(text)

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        chunk = self.data[start_idx : start_idx + self.block_size + 1]
        chunk = torch.tensor(chunk, dtype=torch.long)
        return chunk[:-1], chunk[1:]

def train():
    # --- 設定 ---
    EPOCHS = 50           # 徹底的に覚えさせる
    BATCH_SIZE = 8
    LR = 1e-3             # 少し高めの学習率
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = "models/checkpoints/trained_brain_v20.pth"
    
    logger.info(f"🚀 Starting Overfit Training on {DEVICE}...")

    # モデル構築
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = BitSpikeMamba(
        vocab_size=tokenizer.vocab_size,
        d_model=128, d_state=32, num_layers=4, d_conv=4, expand=2, time_steps=4,
        neuron_config={"type": "lif", "tau_mem": 2.0}
    ).to(DEVICE)

    # データセット
    dataset = ShakespeareDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 最適化
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    try:
        start_total = time.time()
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                
                # SNN Forward
                logits, _, _ = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            
            # 進捗表示 (Lossが下がっていくのを確認してください)
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f}")
            
            # Lossが0.5を切ったら早期終了してもOK
            if avg_loss < 0.1:
                logger.info("🎯 Perfect convergence reached!")
                break
                
        # 保存
        torch.save(model.state_dict(), SAVE_PATH)
        logger.info(f"💾 Trained Brain saved! (Total time: {time.time()-start_total:.1f}s)")
        
    except KeyboardInterrupt:
        logger.warning("Stopped by user.")
        torch.save(model.state_dict(), SAVE_PATH)

def train_demo(epochs=1, max_steps=None):
    # モデル初期化などのロジック
    # ...
    
    for epoch in range(epochs):
        # 進行状況がわかるように tqdm や print を追加
        print(f"  Epoch {epoch+1}/{epochs} starting...")
        
        for i, batch in enumerate(dataloader):
            if max_steps and i >= max_steps:
                break
            # 学習処理...
            
    print("  Training demo finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=5) # デフォルトを小さく設定
    args = parser.parse_args()
    
    train_demo(epochs=args.epochs, max_steps=args.max_steps)
