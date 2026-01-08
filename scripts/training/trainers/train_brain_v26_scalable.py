# ファイルパス: scripts/training/trainers/train_brain_v26_scalable.py
# 日本語タイトル: Brain v2.6 Scalable Trainer (Mypy Fixed)
# 目的: 学習データ量を以前の成功時(v25_fast)と同等以上に増やし、確実に収束させる。

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# プロジェクトルート設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# デッドロック回避
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', stream=sys.stdout, force=True)
logger = logging.getLogger("BrainScalable")

# ★修正: インポートエラー回避。spikingjellyを直接使用。
try:
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
    from spikingjelly.activation_based import functional
except ImportError:
    # ローカル環境等でパスが解決できない場合のフォールバック（mypy用には無視させる）
    pass

class JsonConversationalDataset(Dataset):
    def __init__(self, tokenizer, json_path, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Data file not found: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
            
        logger.info(f"📚 Loading {len(conversations)} conversation pairs from {json_path}...")
        
        # データ量が少ない場合は徹底的に繰り返して「暗記」させる
        if len(conversations) < 20:
            repeat_factor = 150
        elif len(conversations) < 100:
            repeat_factor = 50
        else:
            repeat_factor = 10
            
        logger.info(f"🔄 Data Augmentation: Repeating dataset {repeat_factor} times to ensure convergence.")
        
        for _ in range(repeat_factor):
            for item in conversations:
                user_text = item.get("user", "")
                brain_text = item.get("brain", "")
                
                # プロンプト形式の統一
                full_text = f"User: {user_text}\nBrain: {brain_text}"
                
                ids = tokenizer.encode(full_text, add_special_tokens=True)
                ids.append(tokenizer.eos_token_id)
                self.data.extend(ids)
                
    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        # パディング処理
        chunk = torch.tensor(self.data[start:end], dtype=torch.long)
        if len(chunk) < self.block_size + 1:
            pad = torch.full((self.block_size + 1 - len(chunk),), self.tokenizer.eos_token_id, dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        return chunk[:-1], chunk[1:]

def train():
    # --- Config ---
    CONFIG = {
        "data_path": "data/training_data.json",
        "save_path": "models/checkpoints/trained_brain_v25_fast.pth", # 読み込み側と合わせるため同じパス
        "d_model": 256,
        "num_layers": 4,
        "time_steps": 4,  # 高速化維持
        "batch_size": 8,
        "epochs": 15,     # データが増えたので15エポックで十分収束するはず
        "lr": 2e-3        # 学習率を少し強めに維持
    }

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    print(f"\n>>> 🚀 Starting Boosted Scalable Training on {device.upper()}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = BitSpikeMamba(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        d_state=16, d_conv=4, expand=2,
        num_layers=CONFIG["num_layers"],
        time_steps=CONFIG["time_steps"],
        neuron_config={"type": "lif", "tau_mem": 2.0}
    ).to(device)
    
    # Dataset
    dataset = JsonConversationalDataset(tokenizer, CONFIG["data_path"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    print(">>> Training Loop Started...", flush=True)

    model.train()
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            functional.reset_net(model)
            
            logits, _, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"   Avg Loss: {avg_loss:.4f}")

    # 保存
    torch.save(model.state_dict(), CONFIG["save_path"])
    print(f"\n>>> ✅ Model updated from {CONFIG['data_path']}")

if __name__ == "__main__":
    train()