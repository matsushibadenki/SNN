# ファイルパス: scripts/experiments/brain/train_newton_concept.py
# 日本語タイトル: Brain v4.0 Newton's Training (Visual-Concept Alignment)
# 目的: 「りんごが落ちる映像」と「Gravity」という単語を結びつける短期集中学習を行う。

from snn_research.models.experimental.brain_v4 import SynestheticBrain
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../..")))


def train_newton():
    print("\n=======================================================")
    print(" 🍎 Newton's Training: Teaching 'Gravity' from Vision")
    print("=======================================================\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # 1. モデル準備
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    brain = SynestheticBrain(
        vocab_size=len(tokenizer),
        d_model=256,
        time_steps=8,
        device=device
    ).to(device)

    # 学習対象: 視覚プロジェクター（目と脳の接続部）と脳のコア
    # これにより「映像」を「言葉の意味」に翻訳する回路が形成されます
    optimizer = optim.AdamW(brain.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 2. 教師データの作成
    # 入力テキスト: "The apple fell from the tree due to"
    text_prompt = "The apple fell from the tree due to"
    target_word = " gravity"

    # トークン化
    inputs = tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)
    target_id = tokenizer.encode(target_word, return_tensors="pt")[0, 0].item()
    target_tensor = torch.tensor([target_id], device=device)  # 正解ラベル

    # 映像データ (落下運動)
    image_seq = torch.zeros((1, 8, 1, 28, 28)).to(device)
    for t in range(8):
        y_pos = 2 + t * 3
        if y_pos < 28:
            image_seq[0, t, 0, y_pos:y_pos+3, 12:15] = 1.0

    print(f"📖 Context: '{text_prompt}'")
    print("🎬 Vision:  [Falling Apple Video]")
    print(f"🎯 Target:  '{target_word.strip()}'\n")

    # 3. 学習ループ (Overfitting Demo)
    brain.train()
    print(">>> Starting Training (Simulating 'Aha!' moment)...")

    for epoch in range(51):  # 50回ほど見せて教える
        optimizer.zero_grad()

        # 思考 (Forward)
        logits = brain(text_input=inputs, image_input=image_seq)

        # 予測 (最後のトークンの次の単語)
        last_token_logits = logits[:, -1, :]
        loss = criterion(last_token_logits, target_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # 現在の予測を確認
            pred_id = torch.argmax(last_token_logits, dim=-1).item()
            pred_word = tokenizer.decode([pred_id]).strip()
            print(
                f"   Epoch {epoch:02d}: Loss = {loss.item():.4f} | Brain thinks: '{pred_word}'")

            if pred_word.lower() == "gravity":
                print(
                    f"\n✨ Aha! The brain understood the concept at Epoch {epoch}!")
                break

    # 4. 最終テスト
    print("\n-------------------------------------------------------")
    print(" 🧠 Final Inference Test")
    print("-------------------------------------------------------")
    brain.eval()
    with torch.no_grad():
        logits = brain(text_input=inputs, image_input=image_seq)
        pred_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        final_word = tokenizer.decode([pred_id])

    print(f"Input: \"{text_prompt}\" + [Falling Image]")
    print(f"Brain Answer: \"{final_word.strip()}\"")

    if "gravity" in final_word.lower():
        print("✅ SUCCESS: The visual phenomenon is now grounded to the symbol 'Gravity'.")
    else:
        print("❌ Failed to learn. More training needed.")


if __name__ == "__main__":
    train_newton()
