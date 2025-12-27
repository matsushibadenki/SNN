# scripts/prepare_distillation_data.py
# 教師モデルからロジットを事前計算し、SNN蒸留用のデータセットを作成するスクリプト
#
# ディレクトリ: scripts/prepare_distillation_data.py
# ファイル名: 蒸留データ準備ツール
# 目的: 教師ANNモデルの推論結果（ロジット）を保存し、SNN学習時の計算コストを削減する。
#
# 変更点:
# - [修正 v4] 蒸留効率向上のため、ロジット保存時に温度スケーリング（Temperature）を適用可能に。
# - [修正 v4] デフォルトの教師モデルとして 'gpt2' を使用する際のトークナイザー処理を安定化。
# - [修正 v4] 保存されるロジットテンソルのデータ型を float16 に変更し、ディスク使用量を削減。

import argparse
import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="蒸留用事前計算データ作成ツール")
    parser.add_argument("--input_file", type=str, required=True, help="入力データ(jsonl)")
    parser.add_argument("--output_dir", type=str, required=True, help="出力ディレクトリ")
    parser.add_argument("--teacher_model", type=str, default="gpt2", help="教師モデル名")
    parser.add_argument("--temperature", type=float, default=2.0, help="蒸留用の温度パラメータ (1.0以上を推奨)")
    parser.add_argument("--max_length", type=int, default=32, help="最大シーケンス長")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 出力パスの準備
    output_path = Path(args.output_dir)
    logits_dir = output_path / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)

    # 教師モデルとトークナイザーのロード
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.teacher_model).to(device)
        model.eval()
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return

    # データの読み込み
    samples = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Starting logits pre-computation for {len(samples)} samples (T={args.temperature})...")

    distillation_metadata = []

    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples)):
            text = sample.get("text", "")
            if not text:
                continue

            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                max_length=args.max_length, 
                padding="max_length", 
                truncation=True
            ).to(device)

            outputs = model(**inputs)
            # ロジットの取得 (Batch, Seq, Vocab) -> (Seq, Vocab)
            logits = outputs.logits.squeeze(0)
            
            # [修正] 温度スケーリングの適用
            # これによりソフトターゲットの分布がなだらかになり、SNNの学習が安定します
            logits = logits / args.temperature

            # ファイル保存 (float16に変換して軽量化)
            logit_filename = f"sample_{i}.pt"
            torch.save(logits.cpu().half(), logits_dir / logit_filename)

            # メタデータの作成
            sample["logits_path"] = str(logits_dir / logit_filename)
            distillation_metadata.append(sample)

    # jsonlとして保存
    meta_file = output_path / "distillation_data.jsonl"
    with open(meta_file, "w", encoding="utf-8") as f:
        for entry in distillation_metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Logits pre-computation complete.")
    print(f"   - Metadata saved to: {meta_file}")
    print(f"   - Logit tensors saved in: {logits_dir} (Scaled by T={args.temperature})")

if __name__ == "__main__":
    main()