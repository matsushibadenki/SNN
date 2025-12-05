# ファイルパス: snn_research/data/datasets.py
# (Phase 3: Multimodal Dataset Support - Classification Fix)
# Title: データセット定義
# Description:
# - テキスト、対話、指示データセットに加え、画像-テキストペアを扱う
#   ImageTextDataset を追加。
# - 画像の読み込みと前処理を行う。
# - 修正: ImageTextDatasetで 'label' キーが存在する場合は分類タスクとして扱い、
#   スカラーラベルを返すように変更。

import torch
from torch.utils.data import Dataset
from typing import Iterator, Dict, Any, Tuple, Optional, List, Union
import os
import json
from enum import Enum
from transformers import PreTrainedTokenizerBase
try:
    from PIL import Image
    from torchvision import transforms # type: ignore
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# --- データローダーとデータ形式 ---
class DataFormat(Enum):
    SIMPLE_TEXT = "simple_text"
    DIALOGUE = "dialogue"
    INSTRUCTION = "instruction"
    IMAGE_TEXT = "image_text" # 【追加】

def load_jsonl_data(file_path: str) -> Iterator[Dict[str, Any]]:
    """JSONLファイルを一行ずつ読み込むジェネレータ"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# --- データセットクラス ---
class SNNBaseDataset(Dataset):
    """大規模なJSONLファイルをメモリ効率良く扱うための基底クラス"""
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        self.offsets = []
        with open(self.file_path, 'rb') as f:
            self.offsets.append(f.tell())
            while f.readline():
                self.offsets.append(f.tell())
        self.offsets.pop() 

    def __len__(self):
        return len(self.offsets)

    def _get_json_item(self, idx: int) -> Dict[str, Any]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            return json.loads(line)

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            f"{self.tokenizer.bos_token or ''}{text}",
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors="pt"
        )

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        raise NotImplementedError

# ... (SimpleTextDataset, DialogueDataset, InstructionDataset は変更なし) ...

class SimpleTextDataset(SNNBaseDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self._get_json_item(idx)
        tokenized = self._encode_text(item['text'])
        input_ids = tokenized['input_ids'].squeeze(0)
        return input_ids[:-1], input_ids[1:]

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path): yield item['text']

class DialogueDataset(SNNBaseDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self._get_json_item(idx)
        eos_token = self.tokenizer.eos_token or ''
        full_conversation = f" {eos_token} ".join([turn['value'] for turn in item['conversations']])
        tokenized = self._encode_text(full_conversation)
        input_ids = tokenized['input_ids'].squeeze(0)
        return input_ids[:-1], input_ids[1:]

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path):
            for turn in item['conversations']: yield turn['value']

class InstructionDataset(SNNBaseDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self._get_json_item(idx)
        prompt = item['instruction']
        if 'input' in item and item['input']: prompt += f"\n{item['input']}"
        full_text = f"{prompt}\n{item['output']}"
        tokenized = self._encode_text(full_text)
        input_ids = tokenized['input_ids'].squeeze(0)
        return input_ids[:-1], input_ids[1:]

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path):
            yield item['instruction']
            if 'input' in item and item['input']: yield item['input']
            yield item['output']

# --- 【追加】ImageTextDataset ---
class ImageTextDataset(SNNBaseDataset):
    """
    画像パスとキャプション（テキスト）のペア、または画像とクラスラベルを読み込むデータセット。
    {'image': 'path/to/img.jpg', 'text': '...', 'label': 0}
    """
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerBase, max_seq_len: int, image_size: int = 224):
        super().__init__(file_path, tokenizer, max_seq_len)
        if not PIL_AVAILABLE:
            raise ImportError("ImageTextDataset requires 'Pillow' and 'torchvision'. Please install them.")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # ImageNet mean/std (一般的な正規化)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.root_dir = os.path.dirname(file_path)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._get_json_item(idx)
        
        # 画像処理
        image_path = item.get('image')
        if image_path:
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.root_dir, image_path)
            
            try:
                image = Image.open(image_path).convert('RGB')
                pixel_values = self.transform(image)
            except Exception as e:
                print(f"Warning: Could not load image {image_path}: {e}")
                pixel_values = torch.zeros(3, 224, 224)
        else:
            pixel_values = torch.zeros(3, 224, 224)

        # --- ▼ 修正: ラベルとテキストの処理分岐 ▼ ---
        # 'label' キーがあり、かつ整数型の場合は分類タスクとして扱う
        if 'label' in item and isinstance(item['label'], int):
            labels = torch.tensor(item['label'], dtype=torch.long)
            
            # input_ids は必須ではないが、collate_fn の整合性のためにダミーまたはテキストがあればそれを返す
            if 'text' in item:
                tokenized = self._encode_text(item['text'])
                input_ids = tokenized['input_ids'].squeeze(0)
            else:
                input_ids = torch.tensor([0], dtype=torch.long) # ダミー
                
            return {
                "input_ids": input_ids,
                "labels": labels, # スカラー
                "pixel_values": pixel_values
            }
        else:
            # 'label' がない場合はキャプション生成（Next Token Prediction）として扱う
            text = item.get('text', "")
            tokenized = self._encode_text(text)
            full_input_ids = tokenized['input_ids'].squeeze(0)
            
            return {
                "input_ids": full_input_ids[:-1],
                "labels": full_input_ids[1:], # シーケンス
                "pixel_values": pixel_values
            }
        # --- ▲ 修正 ▲ ---

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path):
            yield item.get('text', "")

class DistillationDataset(SNNBaseDataset):
    # ... (変更なし) ...
    def __init__(self, file_path: str, data_dir: str, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        super().__init__(file_path, tokenizer, max_seq_len)
        self.data_dir = data_dir

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self._get_json_item(idx)
        
        tokenized = self._encode_text(item['text'])
        input_ids = tokenized['input_ids'].squeeze(0)
        
        student_input = input_ids[:-1]
        student_target = input_ids[1:]
        
        logits_path = os.path.join(self.data_dir, item['logits_path'])
        teacher_logits = torch.load(logits_path).to(torch.float32)

        min_len = min(student_input.size(0), teacher_logits.size(0))
        
        student_input = student_input[:min_len]
        student_target = student_target[:min_len]
        teacher_logits = teacher_logits[:min_len]
        
        return student_input, student_target, teacher_logits

def get_dataset_class(data_format: DataFormat) -> type[SNNBaseDataset]:
    format_map = {
        DataFormat.SIMPLE_TEXT: SimpleTextDataset,
        DataFormat.DIALOGUE: DialogueDataset,
        DataFormat.INSTRUCTION: InstructionDataset,
        DataFormat.IMAGE_TEXT: ImageTextDataset # 【追加】
    }
    return format_map[data_format]
