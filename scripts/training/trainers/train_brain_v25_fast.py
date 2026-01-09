# ファイルパス: scripts/training/trainers/train_brain_v25_fast.py
# 日本語タイトル: Brain v2.5 Fast Trainer (Optimized for Mac/MPS)
# 目的:
#   Mac(MPS)でも高速に収束させるための軽量設定版トレーナー。
#   time_stepsを4に削減し、学習率を高めに設定して強制的に会話を覚えさせる。

from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import logging
import os
import sys

# デッドロック回避設定
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# プロジェクトルート設定
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../..")))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("BrainFastTrainer")

try:
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
    from snn_research.activation_based import functional  # type: ignore
except ImportError:
    try:
        from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
        from spikingjelly.activation_based import functional
    except Exception as e:
        logger.error(f"Import Error: {e}")
        sys.exit(1)

# --- 1. Dataset ---


class FastConversationalDataset(Dataset):
    def __init__(self, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # 覚えさせる会話リスト (少数精鋭)
        self.conversations = [
            ("Hello.", "Hi! I am Brain v2.5."),
            ("Who are you?", "I am a Spiking Neural Network AI."),
            ("What is SNN?", "SNN simulates the human brain's neural spikes."),
            ("こんにちは", "こんにちは！調子はどうですか？"),
            ("あなたは誰？", "私はBrain v2.5、AIアシスタントです。"),
            ("元気？", "はい、システムは正常です。"),
            ("Python書いて", "承知しました。\n```python\nprint('Hello SNN')\n```"),
            ("終了", "さようなら。"),
            # System 2 Trigger
            ("Calculate sum.", "Thinking... The sum is calculated by adding numbers."),
            ("フィボナッチ", "思考中... フィボナッチ数列を計算します。")
        ]

        self.data = []
        # データ量を確保 (Epochあたりのステップ数を稼ぐ)
        for _ in range(100):
            for user_text, brain_text in self.conversations:
                full_text = f"User: {user_text}\nBrain: {brain_text}"
                ids = tokenizer.encode(full_text, add_special_tokens=True)
                ids.append(tokenizer.eos_token_id)
                self.data.extend(ids)

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        if end > len(self.data):
            chunk = torch.tensor(self.data[start:], dtype=torch.long)
            pad_len = self.block_size + 1 - len(chunk)
            chunk = torch.cat(
                [chunk, torch.full((pad_len,), self.tokenizer.eos_token_id, dtype=torch.long)])
        else:
            chunk = torch.tensor(self.data[start:end], dtype=torch.long)
        return chunk[:-1], chunk[1:]

# --- 2. Training ---


def train():
    # ★高速化設定★
    CONFIG = {
        "d_model": 256,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 4,      # 6 -> 4層へ軽量化
        "time_steps": 4,      # ★重要: 16 -> 4 (4倍速)
        "neuron_config": {"type": "lif", "tau_mem": 2.0},

        "batch_size": 8,      # バッチサイズ増
        "epochs": 15,         # 短期集中
        "lr": 2e-3,           # 学習率強め
        "save_path": "models/checkpoints/trained_brain_v25_fast.pth"
    }

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    print(f"\n>>> 🚀 Starting Fast Training on {device.upper()}", flush=True)
    print(
        f">>> Configuration: Layers={CONFIG['num_layers']}, TimeSteps={CONFIG['time_steps']}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = BitSpikeMamba(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        d_state=CONFIG["d_state"],
        d_conv=CONFIG["d_conv"],
        expand=CONFIG["expand"],
        num_layers=CONFIG["num_layers"],
        time_steps=CONFIG["time_steps"],
        neuron_config=CONFIG["neuron_config"]
    ).to(device)

    model.train()

    dataset = FastConversationalDataset(tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)

    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"])

    print(">>> Training Loop Started...", flush=True)
    if device == "mps":
        print("    (Note: First batch compilation might take ~30-60 seconds.)", flush=True)

    for epoch in range(CONFIG["epochs"]):
        total_loss = 0.0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            functional.reset_net(model)

            logits, _, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)),
                             targets.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"   Avg Loss: {avg_loss:.4f}", flush=True)

        # 簡易生成チェック
        if (epoch + 1) % 5 == 0:
            generate_sample(model, tokenizer, device, "User: Hello.\nBrain:")

    # 保存
    os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
    state_dict = {k: v for k, v in model.state_dict(
    ).items() if "spikes" not in k and "mem" not in k}
    torch.save(state_dict, CONFIG["save_path"])
    print(f"\n>>> ✅ Model saved to {CONFIG['save_path']}")


def generate_sample(model, tokenizer, device, prompt):
    print("   [Sampling...]", flush=True)
    model.eval()
    try:
        with torch.no_grad():
            functional.reset_net(model)
            curr = tokenizer.encode(prompt, return_tensors='pt').to(device)
            out_tokens = []
            for _ in range(10):
                functional.reset_net(model)
                logits, _, _ = model(curr)
                next_tk = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                if next_tk.item() == tokenizer.eos_token_id:
                    break
                out_tokens.append(next_tk.item())
                curr = torch.cat([curr, next_tk], dim=1)
            print(f"   Brain: {tokenizer.decode(out_tokens)}", flush=True)
    except Exception:
        pass
    model.train()


if __name__ == "__main__":
    train()
