import os
import torch
import numpy as np
import random
import logging
import json
import yaml
import gradio as gr # type: ignore[import-untyped]
from typing import Callable, Iterator, Tuple, List, Optional, Any, Dict, Union
from pathlib import Path
from datetime import datetime
from transformers import PreTrainedTokenizerBase

# -----------------------------------------------------------------------------
# Path: app/utils.py
# Title: SNNプロジェクト 共通ユーティリティ (System Utilities)
# Description: デバイス管理、ロギング、Gradio UI、データ処理(collate_fn)などの共通機能。
#              Mac (MPS) 環境でのSpikingJellyのログ抑制機能を含みます。
# -----------------------------------------------------------------------------

# --- Logging Setup for SpikingJelly Noise Reduction ---
# SpikingJellyはデフォルトでINFOレベルでcupyの不在を警告するため、WARNING以上に引き上げます。
logging.getLogger('spikingjelly').setLevel(logging.WARNING)
logging.getLogger('spikingjelly.activation_based.base').setLevel(logging.WARNING)
logging.getLogger('spikingjelly.activation_based.auto_cuda').setLevel(logging.WARNING)

# --- Device & Reproducibility ---

def get_device(use_cuda: bool = True) -> torch.device:
    """
    利用可能な最適なデバイスを取得します。
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# 互換性のためのエイリアス・ラッパー (既存コードが get_auto_device を呼んでいるため)
def get_auto_device() -> str:
    """実行環境に最適なデバイス名を文字列で返す (互換性維持用)。"""
    device = get_device()
    return device.type

def set_all_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """
    全ての乱数生成器のシードを固定し、学習の再現性を確保する。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"🔒 All seeds set to {seed} (Deterministic mode: ON)")
    else:
        print(f"🔒 All seeds set to {seed} (Deterministic mode: OFF/CPU/MPS)")

# set_seed もエイリアスとして用意
set_seed = set_all_seeds

# --- File I/O Utils (Added for Health Check) ---

def load_config(config_path: str) -> Dict[str, Any]:
    """YAML設定ファイルを読み込みます。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")

def save_json(data: Any, path: str) -> None:
    """データをJSONとして保存します。"""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def setup_logging(log_dir: str = "logs", log_name: str = "app.log", level: int = logging.INFO) -> logging.Logger:
    """ロギングを設定します。"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_name
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True # 既存の設定を上書き
    )
    
    # ライブラリのノイズ抑制
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return logging.getLogger("SNN_Project")

# --- Gradio & Data Processing (Restored from Original) ---

def get_avatar_svgs():
    """
    Gradioチャットボット用のアバターSVGアイコンのタプルを返す。
    """
    user_avatar_svg = r"""
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-user"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
    """
    assistant_avatar_svg = r"""
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-zap"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
    """
    return user_avatar_svg, assistant_avatar_svg

def collate_fn(tokenizer: PreTrainedTokenizerBase, is_distillation: bool) -> Callable[[List[Any]], Any]:
    """
    データローダー用の Collate 関数。
    テキスト(input_ids) と 画像(pixel_values) の両方をバッチ化する。
    """
    def collate(batch: List[Any]) -> Any:
        padding_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        inputs: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        images: List[torch.Tensor] = [] 
        logits: List[torch.Tensor] = []

        for item in batch:
            if isinstance(item, dict):
                inp = item.get('input_ids')
                tgt = item.get('labels')
                img = item.get('pixel_values')

                if inp is not None:
                    inputs.append(torch.tensor(inp) if not isinstance(inp, torch.Tensor) else inp)
                if tgt is not None:
                    targets.append(torch.tensor(tgt) if not isinstance(tgt, torch.Tensor) else tgt)
                if img is not None:
                    images.append(torch.tensor(img) if not isinstance(img, torch.Tensor) else img)

                if is_distillation:
                    lg = item.get('teacher_logits')
                    if lg is not None: logits.append(torch.tensor(lg) if not isinstance(lg, torch.Tensor) else lg)
                    else: logits.append(torch.empty(0))

            elif isinstance(item, tuple) and len(item) >= 2:
                inp = item[0]
                tgt = item[1]
                inputs.append(torch.tensor(inp) if not isinstance(inp, torch.Tensor) else inp)
                targets.append(torch.tensor(tgt) if not isinstance(tgt, torch.Tensor) else tgt)
                if is_distillation:
                    if len(item) >= 3:
                         lg = item[2]
                         if lg is not None: logits.append(torch.tensor(lg) if not isinstance(lg, torch.Tensor) else lg)
                         else: logits.append(torch.empty(0))
                    else: logits.append(torch.empty(0))
            else:
                print(f"Warning: Skipping unsupported batch item type: {type(item)}")
                continue

        if not inputs: return {}

        batch_data: Dict[str, Any] = {}
        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_val)
        
        if targets and targets[0].ndim == 0:
            padded_targets = torch.stack(targets, dim=0)
        else:
            padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)

        attention_mask = torch.ones_like(padded_inputs)
        attention_mask[padded_inputs == padding_val] = 0
        
        batch_data["input_ids"] = padded_inputs
        batch_data["attention_mask"] = attention_mask
        batch_data["labels"] = padded_targets

        if images:
            batch_data["input_images"] = torch.stack(images, dim=0)

        if is_distillation:
            padded_logits = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=0.0)
            if padded_targets.ndim > 1:
                seq_len = padded_inputs.shape[1]
                if padded_targets.shape[1] < seq_len:
                    pad = torch.full((padded_targets.shape[0], seq_len - padded_targets.shape[1]), -100, dtype=padded_targets.dtype, device=padded_targets.device)
                    padded_targets = torch.cat([padded_targets, pad], dim=1)
                if padded_logits.shape[1] < seq_len:
                    pad = torch.full((padded_logits.shape[0], seq_len - padded_logits.shape[1], padded_logits.shape[2]), 0.0, dtype=padded_logits.dtype, device=padded_logits.device)
                    padded_logits = torch.cat([padded_logits, pad], dim=1)
            return padded_inputs, attention_mask, padded_targets, padded_logits

        return batch_data
    
    return collate

def build_gradio_ui(
    stream_fn: Callable[[str, List[List[Optional[str]]]], Iterator[Tuple[List[List[Optional[str]]], str]]],
    title: str,
    description: str,
    chatbot_label: str,
    theme: gr.themes.Base
) -> gr.Blocks:
    """共通のGradio Blocks UIを構築する"""
    with gr.Blocks(theme=theme) as demo:
        gr.Markdown(f"# {title}\n{description}")
        
        initial_stats_md = """
        **Inference Time:** `N/A`
        **Tokens/Second:** `N/A`
        ---
        **Total Spikes:** `N/A`
        **Spikes/Second:** `N/A`
        """

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label=chatbot_label, height=500)
            with gr.Column(scale=1):
                stats_display = gr.Markdown(value=initial_stats_md, label="📊 Inference Stats")

        with gr.Row():
            msg_textbox = gr.Textbox(
                show_label=False,
                placeholder="メッセージを入力...",
                container=False,
                scale=6,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", scale=1)

        gr.Markdown("<footer><p>© 2025 SNN System Design Project. All rights reserved.</p></footer>")

        def clear_all():
            return [], "", initial_stats_md

        submit_event = msg_textbox.submit(
            fn=stream_fn,
            inputs=[msg_textbox, chatbot],
            outputs=[chatbot, stats_display],
            queue=False
        )
        submit_event.then(fn=lambda: "", inputs=None, outputs=msg_textbox)
        
        button_submit_event = submit_btn.click(
            fn=stream_fn,
            inputs=[msg_textbox, chatbot],
            outputs=[chatbot, stats_display],
            queue=False
        )
        button_submit_event.then(fn=lambda: "", inputs=None, outputs=msg_textbox)

        clear_btn.click(
            fn=clear_all,
            inputs=None,
            outputs=[chatbot, msg_textbox, stats_display],
            queue=False
        )
    
    return demo