# matsushibadenki/snn/scripts/prepare_distillation_data.py
# 知識蒸留のための教師モデルのロジットを事前計算するスクリプト
#
# 機能:
# - 元のデータセットを読み込み、教師モデル（例: gpt2）で推論を実行する。
# - 各サンプルに対する教師モデルの出力ロジットを計算する。
# - 元のテキスト情報と、ロジットを保存したファイルへのパスを新しいJSONLファイルに出力する。
# - これにより、学習中の教師モデル推論が不要になり、学習が大幅に高速化される。

import os
import argparse
import json
import torch
from tqdm import tqdm  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset

class RawTextDatasetForLogits(Dataset):
    """単純なテキスト行を読み込むためのデータセット"""
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            # sample_data.jsonlのような形式を想定
            self.lines = [json.loads(line)['text'] for line in f if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

def prepare_distillation_data(args):
    """事前計算を実行するメイン関数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 教師モデルとTokenizerをロード
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.teacher_model).to(device)
    model.eval()

    # データセットとデータローダーを準備
    dataset = RawTextDatasetForLogits(args.input_file)
    
    def collate_fn(batch_texts):
        return tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt"
        )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    logits_dir = os.path.join(args.output_dir, "logits")
    os.makedirs(logits_dir, exist_ok=True)
    
    output_jsonl_path = os.path.join(args.output_dir, "distillation_data.jsonl")

    print(f"Starting logits pre-computation for {len(dataset)} samples...")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Pre-computing logits")):
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                
                # ロジットをCPUに移動し、float16に変換してディスク容量を節約
                logits_cpu = outputs.logits.cpu().to(torch.float16)
                
                # バッチ内の各サンプルのロジットを個別のファイルに保存
                for j in range(logits_cpu.size(0)):
                    original_index = i * args.batch_size + j
                    if original_index < len(dataset):
                        original_text = dataset[original_index]
                        
                        # パディング部分を除いた実際のシーケンス長を取得
                        seq_len = batch['attention_mask'][j].sum().item()
                        
                        # パディング部分を除いたロジットを保存
                        logit_tensor = logits_cpu[j, :seq_len, :]
                        
                        # 保存
                        logit_filename = f"logit_{original_index}.pt"
                        logit_filepath = os.path.join(logits_dir, logit_filename)
                        torch.save(logit_tensor, logit_filepath)
                        
                        # メタデータファイルに書き込み
                        record = {
                            "text": original_text,
                            "logits_path": os.path.join("logits", logit_filename) # 相対パスで保存
                        }
                        f_out.write(json.dumps(record) + "\n")

    print(f"\n✅ Logits pre-computation complete.")
    print(f"   - Metadata saved to: {output_jsonl_path}")
    print(f"   - Logit tensors saved in: {logits_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute teacher logits for knowledge distillation.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input data file (e.g., sample_data.jsonl).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the pre-computed data.")
    parser.add_argument("--teacher_model", type=str, default="gpt2", help="Name of the Hugging Face teacher model.")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length for tokenization.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    
    args = parser.parse_args()
    prepare_distillation_data(args)
