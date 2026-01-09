# ファイルパス: scripts/experiments/brain/evolve_to_synesthesia_v2.py
# 日本語タイトル: Brain v4.0 Evolution V2 (Mypy Fixed)
# 目的: 既存の会話能力を維持したまま、視覚機能(共感覚)を追加学習させる。
# 修正: Mypyエラー(import-not-found/no-redef)を修正。

from snn_research.models.experimental.brain_v4 import SynestheticBrain
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../..")))

# ★修正: パスを experimental に固定し try-except を削除

# --- データセット ---


class SynesthesiaDataset(Dataset):
    def __init__(self, tokenizer, train=True):
        self.tokenizer = tokenizer
        self.mnist = datasets.MNIST('./data', train=train, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ]))

        self.descriptions = {
            0: "A void, a circle, emptiness.",
            1: "A sharp line, a single pillar, unity.",
            2: "A swan, a curve, duality.",
            3: "Two curves, a wave, trinity.",
            4: "A crossing, a chair, stability.",
            5: "A hook, a spiral, motion.",
            6: "A loop at the bottom, growth.",
            7: "An edge, a cliff, mystery.",
            8: "Infinity, loops, eternity.",
            9: "A loop at the top, completion."
        }

    def __len__(self):
        return len(self.mnist) // 50

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        if isinstance(label, torch.Tensor):
            digit = label.item()
        else:
            digit = label

        img_flat = img.view(-1)
        text = f"Digit {digit}: {self.descriptions[digit]}"

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            max_length=24,
            truncation=True
        )
        input_ids = encoding.input_ids.squeeze(0)

        labels = input_ids.clone()
        if 'attention_mask' in encoding:
            mask = encoding.attention_mask.squeeze(0)
            labels[mask == 0] = -100

        return img_flat, input_ids, labels


def train_synesthesia():
    print("\n=======================================================")
    print(" 🧠 Brain v4.0 Evolution V2: Transfer Learning Mode")
    print("    (Freezing Language Center to preserve chat ability)")
    print("=======================================================\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Running on: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. モデル初期化
    brain = SynestheticBrain(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_layers=4,
        time_steps=4,
        device=device
    ).to(device)

    # 以前の賢い脳（会話モデル）をロード
    base_ckpt = "models/checkpoints/trained_brain_v25_fast.pth"
    if os.path.exists(base_ckpt):
        print(f"📥 Loading base conversational knowledge from {base_ckpt}...")
        base_weights = torch.load(base_ckpt, map_location=device)
        brain.core_brain.load_state_dict(base_weights, strict=False)
        print("✅ Base conversation model loaded.")
    else:
        print("⚠️ Base checkpoint NOT found! Brain will be dumb. Please run train_brain_v26_scalable.py first.")
        return

    # 言語野（core_brain）を凍結
    print("❄️ Freezing Core Brain (Language Center)...")
    for param in brain.core_brain.parameters():
        param.requires_grad = False

    # 視覚野とプロジェクターのみ学習
    trainable_params = list(brain.encoder.parameters()) + \
        list(brain.vision_projector.parameters())

    dataset = SynesthesiaDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = optim.AdamW(trainable_params, lr=2e-3)
    criterion = nn.CrossEntropyLoss()

    print(">>> Starting Visual Cortex Training...")

    brain.train()
    brain.core_brain.eval()

    for epoch in range(5):
        total_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/5")

        for imgs, input_ids, labels in progress:
            imgs = imgs.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = brain(image_input=imgs, text_input=input_ids)

            pred_logits = logits[:, -input_ids.shape[1]:, :]
            shift_logits = pred_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(brain.state_dict(),
               "models/checkpoints/brain_v4_synesthesia.pth")
    print("\n✅ Brain v4.0 Evolved. Synesthetic circuits added to existing knowledge.")

    # --- テスト ---
    print("\n[🧪 Synesthesia Test]")
    brain.eval()

    import random
    test_idx = random.randint(0, len(dataset)-1)
    test_img, _, _ = dataset[test_idx]

    _, raw_label = dataset.mnist[test_idx]
    if isinstance(raw_label, torch.Tensor):
        raw_label = raw_label.item()

    print(f"Showing Image of Digit: {raw_label}")

    gen_ids = brain.generate(
        test_img.unsqueeze(0).to(device),
        start_token_id=tokenizer.encode(
            "Digit", return_tensors="pt")[0, 0].item(),
        max_new_tokens=20
    )
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"Response: Digit{generated_text}")


if __name__ == "__main__":
    train_synesthesia()
