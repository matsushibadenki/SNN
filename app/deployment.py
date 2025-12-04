# matsushibadenki/snn4/snn_research/deployment.py
# Title: SNN推論エンジン
# Description: 訓練済みSNNモデルをロードし、テキスト生成のための推論を実行するクラス。
# BugFix: モデルのパスを絶対パスに解決してからロードすることで、ファイルが見つからない問題を解消。
# BugFix: state_dictのキーから 'model.' プリフィックスを削除し、読み込みエラーを修正。
# BugFix: IndentationErrorの修正。
# BugFix: generateメソッドの max_len が None になる TypeError を修正 (最終対策 v5)。
#
# 改善 (v6):
# - doc/SNN開発：基本設計思想.md (セクション6.1, 引用[16]) に基づき、
#   動的推論（SNN Cutoff）を generate メソッドに実装。
#   確信度が閾値を超えた場合に早期終了するロジックを追加。

import torch
import torch.nn.functional as F # ◾️◾️◾️ 追加 ◾️◾️◾️
import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
# typing から time をインポート
from typing import Iterator, Optional, Dict, Any, List, Union, Tuple
import time # time をインポート
from omegaconf import DictConfig, OmegaConf
from snn_research.core.snn_core import SNNCore # SNNCoreをインポート
import logging # logging をインポート

# logging を設定 (すでにある場合は不要)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_auto_device() -> str:
    """実行環境に最適なデバイスを自動的に選択する。"""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

class SNNInferenceEngine:
    """
    学習済みSNNモデルをロードして推論を実行するエンジン。
    """
    def __init__(self, config: DictConfig):
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config
        device_str = OmegaConf.select(config, "device", default="auto") # selectで安全に取得
        self.device = get_auto_device() if device_str == "auto" else device_str

        # ◾️◾️◾️ 追加: SNN Cutoff 設定 ◾️◾️◾️
        self.cutoff_enabled: bool = OmegaConf.select(config, "deployment.cutoff.enabled", default=True)
        self.cutoff_threshold: float = OmegaConf.select(config, "deployment.cutoff.threshold", default=0.95)
        self.cutoff_min_steps: int = OmegaConf.select(config, "deployment.cutoff.min_steps", default=5)
        if self.cutoff_enabled:
            logger.info(f"⚡️ 動的推論 (SNN Cutoff) が有効です (閾値: {self.cutoff_threshold}, 最小ステップ: {self.cutoff_min_steps})")
        # ◾️◾️◾️ ここまで ◾️◾️◾️

        self.last_inference_stats: Dict[str, Any] = {}

        # 先にTokenizerをロードしてvocab_sizeを取得
        tokenizer_path = OmegaConf.select(config, "data.tokenizer_name", default="gpt2") # selectで安全に取得
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Could not load tokenizer from {tokenizer_path}. Error: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # vocab_sizeを渡してSNNCoreを初期化
        vocab_size = len(self.tokenizer)
        model_config = OmegaConf.select(config, "model", default=config) # selectで安全に取得
        if model_config is None:
             logger.warning("Model config is None in SNNInferenceEngine init. Using empty config.")
             model_config = OmegaConf.create({}) # 空のDictConfigを作成
        self.model = SNNCore(model_config, vocab_size=vocab_size)

        model_path_str = OmegaConf.select(config, "model.path", default=None) # selectで安全に取得

        if model_path_str:
            model_path = Path(model_path_str).resolve()

            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    state_dict_to_load: Dict[str, Any]
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                         state_dict_to_load = checkpoint['model_state_dict']
                    elif isinstance(checkpoint, dict): # state_dict そのものの場合
                         state_dict_to_load = checkpoint
                    else:
                         raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

                    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict_to_load.items()}

                    missing_keys, unexpected_keys = self.model.model.load_state_dict(new_state_dict, strict=False)
                    if missing_keys: logger.warning(f"Missing keys when loading state dict: {missing_keys}")
                    if unexpected_keys: logger.warning(f"Unexpected keys when loading state dict: {unexpected_keys}")

                    logger.info(f"✅ Model loaded from {model_path}")
                except RuntimeError as e:
                    logger.warning(f"⚠️ Warning: Failed to load state_dict, possibly due to architecture mismatch: {e}. Using an untrained model.")
                except Exception as e:
                     logger.warning(f"⚠️ Warning: An error occurred while loading model state_dict: {e}. Using an untrained model.")
            else:
                 logger.warning(f"⚠️ Warning: Model file not found at {model_path}. Using an untrained model.")
        else:
            logger.warning("⚠️ Warning: No model path specified in config. Using an untrained model.")


        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_len: Optional[int], stop_sequences: Optional[List[str]] = None) -> Iterator[Tuple[str, Dict[str, float]]]:
        """
        プロンプトに基づいてテキストと統計情報をストリーミング生成する。
        max_len が None の場合にデフォルト値を設定する。
        SNN Cutoffロジックを実装。
        """
        if max_len is None:
            default_max_len = 100 
            logger.warning(f"max_len was None, using default value: {default_max_len}")
            max_len = default_max_len
        elif not isinstance(max_len, int) or max_len <= 0:
             default_max_len = 100
             logger.warning(f"Invalid max_len value ({max_len}), using default value: {default_max_len}")
             max_len = default_max_len

        tokenizer_callable = getattr(self.tokenizer, "__call__", None)
        if not callable(tokenizer_callable):
            raise TypeError("Tokenizer is not callable.")
        input_ids = tokenizer_callable(prompt, return_tensors="pt")["input_ids"].to(self.device)

        total_spikes = 0.0
        start_time = time.time() 

        for i in range(max_len):
            loop_start_time = time.time() 
            with torch.no_grad():
                outputs, avg_spikes_tensor, _ = self.model(input_ids, return_spikes=True)

            avg_spikes = avg_spikes_tensor.item() if isinstance(avg_spikes_tensor, torch.Tensor) else 0.0
            total_spikes += avg_spikes * input_ids.shape[1] 

            next_token_logits = outputs[:, -1, :]
            
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓SNN Cutoff 実装↓◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            # 確信度（Softmaxの最大値）を計算
            probabilities = F.softmax(next_token_logits, dim=-1)
            confidence, next_token_id = torch.max(probabilities, dim=-1)
            next_token_id = next_token_id.unsqueeze(-1)

            if self.cutoff_enabled and confidence.item() > self.cutoff_threshold and i >= self.cutoff_min_steps:
                logger.info(f"⚡️ SNN Cutoff発動: ステップ {i} で確信度 {confidence.item():.2%} が閾値 {self.cutoff_threshold:.2%} を超過。")
                try:
                     new_token = self.tokenizer.decode(next_token_id.item())
                except Exception as e:
                     logger.error(f"Error decoding token ID {next_token_id.item()}: {e}")
                     new_token = "[Decode Error]"
                
                current_duration = time.time() - start_time
                current_stats = {"total_spikes": total_spikes, "cutoff_step": i}
                yield new_token, current_stats
                break # Cutoffによりループを早期終了
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑SNN Cutoff 実装↑◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

            if next_token_id.item() == getattr(self.tokenizer, 'eos_token_id', None):
                break

            try:
                 new_token = self.tokenizer.decode(next_token_id.item())
            except Exception as e:
                 logger.error(f"Error decoding token ID {next_token_id.item()}: {e}")
                 new_token = "[Decode Error]"

            if stop_sequences and any(seq in new_token for seq in stop_sequences):
                break

            current_duration = time.time() - start_time
            current_stats = {"total_spikes": total_spikes}
            yield new_token, current_stats

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            loop_duration = time.time() - loop_start_time

        self.last_inference_stats = {"total_spikes": total_spikes}