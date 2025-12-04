# matsushibadenki/snn/SNN-2e86a361c76c65c1a2b1c1d1b4046d5ebb98997a/scripts/data_preparation.py
#
# 大規模言語モデル事前学習用データセット (WikiText) の準備スクリプト
#
# 目的:
# - ロードマップ フェーズ1「1.3. データセットの拡充と前処理」に対応。
# - SNNモデルの汎用的な言語能力を向上させるため、大規模な公開コーパスを
#   学習パイプラインに供給する。
#
# 機能:
# 1. Hugging Face `datasets`ライブラリからWikiText-103データセットをダウンロード。
# 2. テキストをクリーニングし、モデルの学習に適した形式に整える。
# 3. 処理後のデータを、`main.py`で扱える`simple_text`形式の.jsonlファイルとして保存。

import os
import json
from datasets import load_dataset  # type: ignore
from tqdm import tqdm  # type: ignore

def prepare_wikitext_data(output_dir: str = "data", cache_dir: str = ".cache"):
    """
    WikiText-103データセットをダウンロードし、クリーニング後に.jsonl形式で保存する。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "wikitext-103_train.jsonl")

    if os.path.exists(output_path):
        print(f"✅ Preprocessed data already exists at '{output_path}'. Skipping.")
        return output_path

    print("Downloading WikiText-103 dataset...")
    # 'wikitext-103-raw-v1' を使用
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", cache_dir=cache_dir)
    
    print(f"Processing and cleaning the dataset -> {output_path}")
    
    cleaned_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Processing WikiText"):
            text = example['text'].strip()
            
            # --- データ品質を担保する前処理 ---
            # 1. 空行はスキップ
            if not text:
                continue
            
            # 2. 記事のセクションタイトル（例: " = Section Name = "）は除外
            if text.startswith(" = ") and text.endswith(" = "):
                continue
            
            # 3. 短すぎる行（5単語未満）は文脈情報が少ないため除外
            if len(text.split()) < 5:
                continue

            record = {"text": text}
            f.write(json.dumps(record) + "\n")
            cleaned_count += 1
            
    print(f"✅ WikiText data preparation complete. Saved {cleaned_count} lines to '{output_path}'.")
    return output_path

if __name__ == "__main__":
    prepare_wikitext_data()