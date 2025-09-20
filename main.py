# matsushibadenki/snn/main.pyの修正
# SNNモデルの学習と推論を実行するためのメインスクリプト (データ形式仕様書v1.0対応版)
#
# 改善点:
# - データ形式仕様書に基づき、.jsonl形式の読み込みに対応。
# - --data_format引数を導入し、'simple_text', 'dialogue', 'instruction'の形式を動的に切り替え可能に。
# - データセット部分を抽象化し、各形式に対応する専用のDatasetクラスを実装。
# - 語彙構築プロセスを汎用化し、複雑なデータ構造からもテキストを抽出できるように改善。
# - ベンチマークスクリプトから呼び出せるように学習ロジックを関数化。

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import itertools
from typing import List, Tuple, Dict, Any, Iterator
import os
import random
import argparse
import json
from enum import Enum

# snn_coreから主要コンポーネントをインポート
from snn_core import BreakthroughSNN, BreakthroughTrainer, CombinedLoss

# ----------------------------------------
# 1. データ形式とローダー
# ----------------------------------------

class DataFormat(Enum):
    """サポートするデータ形式を定義"""
    SIMPLE_TEXT = "simple_text"
    DIALOGUE = "dialogue"
    INSTRUCTION = "instruction"

def load_jsonl_data(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    JSON Lines (.jsonl) ファイルを1行ずつ遅延読み込みします。

    Args:
        file_path (str): データファイルのパス。

    Yields:
        Dict[str, Any]: ファイル内の各行のJSONオブジェクト。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")

    if not file_path.endswith('.jsonl'):
        print(f"警告: ファイル '{file_path}' は .jsonl 拡張子ではありません。JSON Lines形式として処理を試みます。")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# ----------------------------------------
# 2. 語彙の構築
# ----------------------------------------

class Vocabulary:
    """テキストとIDを相互変換するための語彙クラス"""
    def __init__(self):
        # 予約トークン
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build_vocab(self, all_texts: Iterator[str]):
        all_words = itertools.chain.from_iterable(txt.lower().split() for txt in all_texts)
        word_counts = Counter(all_words)
        for word, _ in word_counts.items():
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, text: str, add_start_end: bool = True) -> List[int]:
        tokens = [self.word2idx.get(word.lower(), self.special_tokens["<UNK>"]) for word in text.split()]
        if add_start_end:
            return [self.special_tokens["<START>"]] + tokens + [self.special_tokens["<END>"]]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        # <START> と <END> トークンはデコード結果から除外することが多い
        ids_to_decode = [idx for idx in token_ids if idx not in (self.special_tokens["<START>"], self.special_tokens["<END>"])]
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in ids_to_decode])

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
    
    @property
    def pad_id(self) -> int:
        return self.special_tokens["<PAD>"]

# ----------------------------------------
# 3. データセットクラス
# ----------------------------------------

class SNNBaseDataset(Dataset):
    """全てのデータセットクラスの基底クラス"""
    def __init__(self, file_path: str, vocab: Vocabulary):
        self.vocab = vocab
        self.data = list(load_jsonl_data(file_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        raise NotImplementedError

class SimpleTextDataset(SNNBaseDataset):
    """ 'simple_text' 形式のデータセット """
    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.vocab.encode(item['text'])
        return torch.tensor(encoded[:-1]), torch.tensor(encoded[1:], dtype=torch.long)

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path):
            yield item['text']

class DialogueDataset(SNNBaseDataset):
    """ 'dialogue' 形式のデータセット """
    def __getitem__(self, idx):
        item = self.data[idx]
        full_conversation = " ".join([turn['value'] for turn in item['conversations']])
        encoded = self.vocab.encode(full_conversation)
        return torch.tensor(encoded[:-1]), torch.tensor(encoded[1:], dtype=torch.long)

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path):
            for turn in item['conversations']:
                yield turn['value']

class InstructionDataset(SNNBaseDataset):
    """ 'instruction' 形式のデータセット """
    def __getitem__(self, idx):
        item = self.data[idx]
        # 指示と入力を結合してプロンプトを作成
        prompt = item['instruction']
        if 'input' in item and item['input']:
            prompt += f"\n{item['input']}"
        
        # プロンプトと出力を結合して完全なテキストを作成
        full_text = f"{prompt}\n{item['output']}"
        encoded = self.vocab.encode(full_text)
        
        # 出力部分のみを損失計算の対象とするため、プロンプト部分は無視する
        # ここではシンプルに全体を学習対象とするが、高度化も可能
        return torch.tensor(encoded[:-1]), torch.tensor(encoded[1:], dtype=torch.long)

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path):
            yield item['instruction']
            if 'input' in item and item['input']:
                yield item['input']
            yield item['output']

def create_dataset(data_format: DataFormat, file_path: str, vocab: Vocabulary) -> SNNBaseDataset:
    """データ形式に応じて適切なデータセットクラスをインスタンス化するファクトリ関数"""
    format_map = {
        DataFormat.SIMPLE_TEXT: SimpleTextDataset,
        DataFormat.DIALOGUE: DialogueDataset,
        DataFormat.INSTRUCTION: InstructionDataset
    }
    # data_formatが文字列で渡される場合も考慮
    data_format_enum = DataFormat(data_format) if isinstance(data_format, str) else data_format
    if data_format_enum not in format_map:
        raise ValueError(f"サポートされていないデータ形式です: {data_format}")
    return format_map[data_format_enum](file_path, vocab)

def get_text_extractor(data_format: DataFormat) -> callable:
    """データ形式に応じたテキスト抽出関数を取得"""
    format_map = {
        DataFormat.SIMPLE_TEXT: SimpleTextDataset.extract_texts,
        DataFormat.DIALOGUE: DialogueDataset.extract_texts,
        DataFormat.INSTRUCTION: InstructionDataset.extract_texts
    }
    # data_formatが文字列で渡される場合も考慮
    data_format_enum = DataFormat(data_format) if isinstance(data_format, str) else data_format
    return format_map[data_format_enum]

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """可変長のシーケンスをパディングするためのCollate関数"""
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_id)
    return padded_inputs, padded_targets

# ----------------------------------------
# 4. 推論エンジン
# ----------------------------------------

class SNNInferenceEngine:
    """SNNモデルでテキスト生成や分析を行う推論エンジン"""
    def __init__(self, model_path: str, device: str = "cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.vocab = checkpoint['vocab']
        config = checkpoint['config']
        
        self.model = BreakthroughSNN(vocab_size=self.vocab.vocab_size, **config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def generate(self, start_text: str, max_len: int = 20) -> str:
        # print(f"\n生成開始: '{start_text}'") # ベンチマーク中はログを抑制
        
        input_ids = self.vocab.encode(start_text, add_start_end=True)[:-1] # ENDトークンは不要
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        generated_ids = list(input_ids)
        
        with torch.no_grad():
            for _ in range(max_len):
                logits, _ = self.model(input_tensor, return_spikes=True) # 修正: 戻り値の形式に合わせる
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                if next_token_id == self.vocab.special_tokens["<END>"]:
                    break
                
                generated_ids.append(next_token_id)
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=self.device)], dim=1)
        
        return self.vocab.decode(generated_ids)

# ----------------------------------------
# 5. 実行ブロック
# ----------------------------------------

def run_training(args: argparse.Namespace) -> Vocabulary:
    """
    モデルの学習を実行し、学習済みの語彙を返す。
    外部スクリプト（ベンチマークなど）からの呼び出しを想定。
    """
    print(f"🚀 革新的SNNシステムの訓練開始 (データ形式: {args.data_format})")
    
    try:
        # データから語彙を構築
        vocab = Vocabulary()
        print("📖 語彙を構築中...")
        text_extractor = get_text_extractor(args.data_format)
        vocab.build_vocab(text_extractor(args.data_path))
        print(f"✅ 語彙を構築しました。語彙数: {vocab.vocab_size}")

        # データセットとデータローダーを作成
        dataset = create_dataset(args.data_format, args.data_path, vocab)
        custom_collate_fn = lambda batch: collate_fn(batch, vocab.pad_id)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        print(f"✅ {args.data_path} から {len(dataset)} 件のデータを読み込みました。")

    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        print(f"❌ エラー: データファイルの読み込みまたは処理に失敗しました。\n詳細: {e}")
        print("ヒント: --data_format 引数がファイルの内容と一致しているか、.jsonl ファイルが仕様書通りか確認してください。")
        raise e

    # モデル設定
    config = {'d_model': args.d_model, 'd_state': args.d_state, 'num_layers': args.num_layers, 'time_steps': args.time_steps}
    model = BreakthroughSNN(vocab_size=vocab.vocab_size, **config)
    
    # 汎用Trainerのセットアップ
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = CombinedLoss()
    trainer = BreakthroughTrainer(model, optimizer, criterion)
    
    # 学習ループ
    print("\n🔥 学習を開始します...")
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(dataloader)
        val_metrics = trainer.evaluate(dataloader) # 評価も追加
        if (epoch + 1) % args.log_interval == 0:
            metrics_str = ", ".join([f"train_{k}: {v:.4f}" for k, v in train_metrics.items()])
            metrics_str += ", " + ", ".join([f"val_{k}: {v:.4f}" for k, v in val_metrics.items()])
            print(f"Epoch {epoch+1: >3}/{args.epochs}: {metrics_str}")
            
    # モデル保存
    model_path = args.model_path
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': config
    }, model_path)
    print(f"\n✅ 学習済みモデルを '{model_path}' に保存しました。")
    return vocab


def start_inference_cli(args):
    """学習済みモデルで推論（文章生成）を実行するためのCLI"""
    try:
        engine = SNNInferenceEngine(model_path=args.model_path)
        
        print("\n💬 テキスト生成を開始します。終了するには 'exit' または 'quit' と入力してください。")
        while True:
            start_text = input("入力テキスト: ")
            if start_text.lower() in ["exit", "quit"]:
                break
            generated_text = engine.generate(start_text, max_len=args.max_len)
            print(f"生成結果: {generated_text}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"エラー: 学習済みモデルファイル({args.model_path})が必要です。先に 'train' コマンドで実行してください。")
    except Exception as e:
        print(f"❌ 予期せぬエラーが発生しました: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNNベース AIチャットシステム (データ形式仕様書v1.0対応)")
    subparsers = parser.add_subparsers(dest="command", required=True, help="実行するコマンド")

    # --- 学習コマンド ---
    parser_train = subparsers.add_parser("train", help="SNNモデルを学習します")
    parser_train.add_argument("data_path", type=str, help="学習データのファイルパス (.jsonl)")
    parser_train.add_argument(
        "--data_format",
        type=DataFormat,
        default=DataFormat.SIMPLE_TEXT,
        choices=list(DataFormat),
        help="学習データの形式"
    )
    parser_train.add_argument("--epochs", type=int, default=100, help="学習エポック数")
    parser_train.add_argument("--batch_size", type=int, default=4, help="バッチサイズ")
    parser_train.add_argument("--learning_rate", type=float, default=5e-4, help="学習率")
    parser_train.add_argument("--log_interval", type=int, default=10, help="ログを表示するエポック間隔")
    parser_train.add_argument("--model_path", type=str, default="breakthrough_snn_model.pth", help="学習済みモデルの保存パス")
    # モデル設定の引数を追加
    parser_train.add_argument("--d_model", type=int, default=64)
    parser_train.add_argument("--d_state", type=int, default=32)
    parser_train.add_argument("--num_layers", type=int, default=2)
    parser_train.add_argument("--time_steps", type=int, default=16)
    
    # --- 推論コマンド ---
    parser_inference = subparsers.add_parser("inference", help="学習済みモデルで推論（テキスト生成）を実行します")
    parser_inference.add_argument("--model_path", type=str, default="breakthrough_snn_model.pth", help="学習済みモデルのパス")
    parser_inference.add_argument("--max_len", type=int, default=40, help="生成するテキストの最大長")

    args = parser.parse_args()

    if args.command == "train":
        run_training(args)
    elif args.command == "inference":
        start_inference_cli(args)
