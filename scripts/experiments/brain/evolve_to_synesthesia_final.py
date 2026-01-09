# ファイルパス: scripts/experiments/brain/evolve_to_synesthesia_final.py
# 日本語タイトル: Brain v4.0 Evolution Final (Mypy Fixed)
# 目的: 凍結された言語野に対して、視覚野を完璧にアライメント（適合）させる。
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

# ★修正: パスを固定

# --- データセット (Augmentation強化) ---


class SynesthesiaDataset(Dataset):
    def __init__(self, tokenizer, train=True):
        self.tokenizer = tokenizer
        # 視覚野を鍛えるため、回転やズームでバリエーションを増やす
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        if train:
            transform_list.insert(0, transforms.RandomRotation(15))
            transform_list.insert(
                0, transforms.RandomAffine(0, translate=(0.1, 0.1)))

        self.mnist = datasets.MNIST('./data', train=train, download=True,
                                    transform=transforms.Compose(transform_list))

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
        return len(self.mnist) // 10  # データ量を増やす

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        if isinstance(label, torch.Tensor):
            digit = label.item()
        else:
            digit = label

        img_flat = img.view(-1)
        text = f"Digit {digit}: {self.descriptions[digit]}"

        encoding = self.tokenizer(
            text, return_tensors='pt', padding='max_length', max_length=24, truncation=True
        )
        input_ids = encoding.input_ids.squeeze(0)
        labels = input_ids.clone()
        if 'attention_mask' in encoding:
            labels[encoding.attention_mask.squeeze(0) == 0] = -100

        return img_flat, input_ids, labels


def train_synesthesia():
    print("\n=======================================================")
    print(" 🧠 Brain v4.0 Final Evolution: Vision Alignment")
    print("=======================================================\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Running on: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. モデル初期化 & 転移学習
    brain = SynestheticBrain(
        vocab_size=tokenizer.vocab_size, d_model=256, num_layers=4, time_steps=4, device=device
    ).to(device)

    base_ckpt = "models/checkpoints/trained_brain_v25_fast.pth"
    if os.path.exists(base_ckpt):
        print("📥 Loading base chat capability...")
        brain.core_brain.load_state_dict(torch.load(
            base_ckpt, map_location=device), strict=False)
    else:
        print("⚠️ Base checkpoint not found. Please train v25 first.")
        return

    # 言語野凍結
    print("❄️ Freezing Language Center.")
    for param in brain.core_brain.parameters():
        param.requires_grad = False

    # 2. オプティマイザ設定 (Projectorを重点的に学習)
    optimizer = optim.AdamW([
        {'params': brain.encoder.parameters(), 'lr': 5e-4},       # 目: 少しゆっくり
        {'params': brain.vision_projector.parameters(), 'lr': 2e-3}  # 接続部: 急速に適合
    ])
    criterion = nn.CrossEntropyLoss()

    dataset = SynesthesiaDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # バッチサイズ倍増

    print(">>> Starting Visual Alignment...")
    brain.train()
    brain.core_brain.eval()  # 言語野は推論モード固定

    for epoch in range(15):  # 15エポック

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/15")

        for imgs, input_ids, labels in progress:
            imgs, input_ids, labels = imgs.to(
                device), input_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = brain(image_input=imgs, text_input=input_ids)

            # Loss計算
            pred_logits = logits[:, -input_ids.shape[1]:, :]
            shift_logits = pred_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))
            loss.backward()
            optimizer.step()

            progress.set_postfix(loss=loss.item())

    # 保存
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(brain.state_dict(),
               "models/checkpoints/brain_v4_synesthesia.pth")
    print("\n✅ Vision aligned with Language Center.")

    # テスト
    brain.eval()
    import random
    test_img, _, _ = dataset[random.randint(0, len(dataset)-1)]
    print("\n[Visual Test]")
    gen_ids = brain.generate(
        test_img.unsqueeze(0).to(device),
        start_token_id=tokenizer.encode(
            "Digit", return_tensors="pt")[0, 0].item(),
        max_new_tokens=20
    )
    print(f"I see: Digit{tokenizer.decode(gen_ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    train_synesthesia()
