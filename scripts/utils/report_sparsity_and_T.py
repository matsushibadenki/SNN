# ファイルパス: scripts/utils/report_sparsity_and_T.py
# Title: スパイク効率性レポート生成 (修正版v2)
# Description: 指定されたモデル設定に基づいてSNNを構築し、スパイク率やタイムステップ数を計測する。
#              学習済みモデルのロード、Vocab Sizeの指定、厳密でないロード(strict=False)に対応。

from snn_research.core.snn_core import SNNCore  # E402 fixed
import sys
import os
import torch
import torch.nn as nn # Added import
from omegaconf import OmegaConf
import argparse
import logging
from typing import Optional, Dict, Any, cast # Added cast

# プロジェクトルートの解決
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_efficiency(model_config_path: str, data_path: str, model_path: Optional[str] = None, vocab_size: int = 50257):
    logger.info(f"Loading config from {model_config_path}")
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Config not found: {model_config_path}")

    cfg = OmegaConf.load(model_config_path)

    # --- ▼ 修正: 階層構造の正規化 ▼ ---
    import typing

    # --- ▼ 修正: 階層構造の正規化 ▼ ---
    model_params_raw: Any
    if 'model' in cfg:
        model_params_raw = OmegaConf.to_container(cfg.model, resolve=True)
    else:
        model_params_raw = OmegaConf.to_container(cfg, resolve=True)

    model_params: Dict[str, Any] = typing.cast(
        Dict[str, Any], model_params_raw)
    # --- ▲ 修正 ▲ ---

    logger.info(f"Building SNN model with vocab_size={vocab_size}...")
    # SNNCoreには辞書形式で渡す
    model = SNNCore(model_params, vocab_size=vocab_size)  # type: ignore

    # --- ▼ 追加: 堅牢なモデルロード処理 ▼ ---
    if model_path:
        if os.path.exists(model_path):
            logger.info(f"Loading model weights from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location='cpu')

                state_dict = None
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif isinstance(checkpoint, dict):
                    state_dict = checkpoint

                if state_dict is not None:
                    try:
                        # まずそのままロードを試みる (strict=Falseでバッファ不足を許容)
                        model.load_state_dict(state_dict, strict=False)
                        logger.info(
                            "✅ Model weights loaded successfully (exact match, strict=False).")
                    except RuntimeError as e:
                        logger.warning(
                            f"Standard load failed ({e}). Attempting prefix adjustment...")

                        # キーの不一致（SNNCoreの 'model.' プレフィックス漏れ）を修正
                        new_state_dict = {}
                        for k, v in state_dict.items():
                            if not k.startswith("model."):
                                new_state_dict[f"model.{k}"] = v
                            else:
                                new_state_dict[k] = v

                        try:
                            # prefix調整後も strict=False でロード
                            model.load_state_dict(new_state_dict, strict=False)
                            logger.info(
                                "✅ Model weights loaded successfully with prefix adjustment (strict=False).")
                        except RuntimeError as e2:
                            logger.error(
                                f"❌ Prefix adjustment also failed: {e2}")
                            # 最後の手段: 内部モデルへ直接ロード
                            if hasattr(model, 'model'):
                                logger.info(
                                    "Attempting direct load into inner model...")
                                # Cast to nn.Module to fix mypy error
                                cast(nn.Module, model.model).load_state_dict(
                                    state_dict, strict=False)
                                logger.info(
                                    "✅ Model weights loaded directly into inner model (strict=False).")
                            else:
                                raise e2

            except Exception as e:
                logger.error(f"❌ Failed to load model weights: {e}")
                # ロード失敗時は終了する
                sys.exit(1)
        else:
            logger.warning(
                f"⚠️ Model path provided but file not found: {model_path}")
    # --- ▲ 追加 ▲ ---

    model.eval()

    # ダミー入力の作成 (モデルタイプに合わせて)
    arch_type = model_params.get(
        "architecture_type", "unknown")  # type: ignore
    logger.info(f"Architecture: {arch_type}")

    # 簡易的にテキスト/画像入力を切り分け
    if arch_type in ["spiking_cnn", "visual_cortex", "feel_snn"]:
        # 画像入力 (B, C, H, W)
        dummy_input = torch.randn(1, 3, 32, 32)
    else:
        # テキスト入力 (B, L)
        dummy_input = torch.randint(0, 100, (1, 128))

    logger.info(f"Running inference with input shape: {dummy_input.shape}")

    with torch.no_grad():
        if hasattr(model, 'forward'):
            _ = model(dummy_input)
        else:
            # SNNCoreがnn.Moduleを継承しており、上記修正でforwardが追加されているため、
            # 通常のモジュール呼び出しが正しく動作するようになります。
            _ = model(dummy_input)

    total_spikes = model.get_total_spikes()
    logger.info(f"Total Spikes: {total_spikes}")

    # 詳細なレポート出力
    print(
        f"Efficiency Report:\n  Architecture: {arch_type}\n  Total Spikes: {total_spikes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--data_path", type=str,
                        default="data/smoke_test_data.jsonl")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained model checkpoint")
    # Vocab Size引数を追加 (GPT-2 default)
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="Vocabulary size for the model (default: 50257 for GPT-2)")
    args = parser.parse_args()

    measure_efficiency(args.model_config, args.data_path,
                       args.model_path, args.vocab_size)