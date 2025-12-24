# ファイルパス: scripts/train_spiking_vlm.py
# (Phase 3: Visual-Language Alignment - Training Script)
# Title: Spiking VLM Training Script (Fix: Architecture Type Added)
# Description:
#   ImageTextDatasetを用いてSpikingVLMを学習させるスクリプト。
#   修正: vision_config と language_config に architecture_type を明示。

import sys
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# プロジェクトルートをPythonパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.models.transformer.spiking_vlm import SpikingVLM
from snn_research.data.datasets import ImageTextDataset, DataFormat

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train SpikingVLM on Image-Text Data")
    
    # データ設定
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL dataset (e.g., data/coco_captions.jsonl)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=64, help="Maximum sequence length for text")
    parser.add_argument("--image_size", type=int, default=224, help="Image resolution")
    
    # モデル設定
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size (default: BERT/DistilBERT)")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--vision_dim", type=int, default=128, help="Vision encoder output dimension")
    parser.add_argument("--time_steps", type=int, default=4, help="SNN time steps")
    parser.add_argument("--use_bitnet", action="store_true", help="Use 1.58bit quantization for projector")
    
    # 学習設定
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/mps/cpu)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/vlm", help="Directory to save checkpoints")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device)
    logger.info(f"🚀 Starting SpikingVLM Training on {device}")

    # 1. トークナイザーの準備
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception:
        logger.warning("Could not load distilbert tokenizer. Using basic whitespace tokenizer fallback.")
        # ダミートークナイザー（依存関係回避用）
        class DummyTokenizer:
            pad_token_id = 0
            bos_token = "[CLS]"
            eos_token = "[SEP]"
            def __call__(self, text, **kwargs):
                # 簡易的なハッシュによるID化
                ids = [hash(w) % args.vocab_size for w in text.split()]
                ids = ids[:args.max_seq_len]
                return {"input_ids": torch.tensor([ids])}
        tokenizer = DummyTokenizer()

    # 2. データセットとデータローダー
    logger.info(f"📂 Loading dataset from {args.data_path}")
    dataset = ImageTextDataset(
        file_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        image_size=args.image_size
    )
    
    def collate_fn(batch):
        # バッチ内のテンソルをスタック
        # エラーハンドリング: Noneが含まれている場合はスキップ
        batch = [b for b in batch if b is not None]
        if not batch:
            return None, None, None
            
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        return input_ids, labels, pixel_values

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    logger.info(f"   - Total samples: {len(dataset)}")
    logger.info(f"   - Batch size: {args.batch_size}")

    # 3. モデル構築
    # 【修正】architecture_type を追加
    vision_config = {
        "architecture_type": "spiking_cnn", # ここが不足していました
        "input_channels": 3,
        "features": args.vision_dim,
        "time_steps": args.time_steps,
        "layers": [64, 128, args.vision_dim] 
    }
    
    language_config = {
        "architecture_type": "spiking_transformer", # ここが不足していました
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "num_layers": 4,
        "num_heads": 4,
        "time_steps": args.time_steps,
        "max_len": args.max_seq_len
    }
    
    projector_config = {
        "visual_dim": args.vision_dim,
        "use_bitnet": args.use_bitnet
    }
    
    model = SpikingVLM(
        vocab_size=args.vocab_size,
        vision_config=vision_config,
        language_config=language_config,
        projector_config=projector_config
    ).to(device)
    
    # 4. オプティマイザと損失関数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)

    # 5. 学習ループ
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batch_count = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for input_ids, labels, pixel_values in progress_bar:
            if input_ids is None: continue
            
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            pixel_values = pixel_values.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            logits, avg_spikes, _ = model(
                input_ids=input_ids,
                input_images=pixel_values,
                return_spikes=True
            )
            
            # Loss計算
            loss = criterion(logits.reshape(-1, args.vocab_size), labels.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Spk": f"{avg_spikes.mean().item():.2f}"})
        
        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            logger.info(f"🏁 Epoch {epoch+1} Completed. Avg Loss: {avg_epoch_loss:.4f}")
            
            # チェックポイント保存
            save_path = os.path.join(args.output_dir, f"spiking_vlm_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"   Saved checkpoint to {save_path}")

    logger.info("🎉 Training Finished Successfully.")

if __name__ == "__main__":
    main()