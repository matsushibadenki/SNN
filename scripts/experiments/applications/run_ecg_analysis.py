# ファイルパス: scripts/experiments/applications/run_ecg_analysis.py
# (SOTAアーキテクチャ対応・Warning抑制版)
# Title: ECG異常検出 デモアプリケーション
# Description:
# - ダミーのECGデータを生成し、SNNモデルで異常/正常の分類を実行するデモ。
# 修正点:
# - モデルパスが提供されていない場合のログレベルを WARNING から INFO に変更。

from snn_research.core.snn_core import SNNCore
from app.utils import get_auto_device
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
import logging
import random
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Tuple, cast, Any, Dict, List

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))


class MockSNNCore(nn.Module):
    """SNNCoreのダミー実装 (フォールバック用)"""

    def __init__(self, num_classes=2, features=128):
        super().__init__()
        self.features = features
        self.fc = nn.Linear(features, num_classes)  # ダミーの出力層
        self.num_classes = num_classes
        self._total_spikes = 0

    def forward(self, x, return_spikes=False, **kwargs):
        batch_size = x.shape[0]
        dummy_features = torch.randn(
            batch_size, self.features, device=x.device)
        logits = self.fc(dummy_features)
        avg_spikes_val = 1500.0 + random.random() * 100
        self._total_spikes += avg_spikes_val * batch_size
        avg_spikes = torch.tensor(avg_spikes_val, device=x.device)
        return logits, avg_spikes, torch.tensor(0.0, device=x.device)

    def get_total_spikes(self) -> float:
        return self._total_spikes

    def reset_spike_stats(self):
        self._total_spikes = 0


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_dummy_ecg_data(batch_size: int, time_steps: int, features: int, device: str) -> torch.Tensor:
    """
    ダミーのECG時系列データを生成する。
    """
    logger.info(
        f"   -> Generating {batch_size} dummy ECG samples ({time_steps} steps, {features} features)...")

    # データを (B, T, F) の形状で生成
    data = torch.randn(batch_size, time_steps, features, device=device) * 0.1

    # 周期的な心拍様パターン (最初の特徴量にのみ追加)
    time_vector = torch.arange(time_steps, device=device).float(
    ) / time_steps * (2 * torch.pi * (time_steps / 1000.0))
    heartbeat_pattern = torch.sin(time_vector).unsqueeze(0).unsqueeze(-1) * 0.5
    data[..., 0] += heartbeat_pattern.squeeze(-1)  # (B, T)

    # 高周波ノイズも少し追加
    data[..., 0] += torch.randn(batch_size, time_steps, device=device) * 0.05

    # 異常スパイク (最初の特徴量にのみ追加)
    for i in range(batch_size):
        if random.random() < 0.2:
            spike_time = random.randint(time_steps // 4, 3 * time_steps // 4)
            spike_height = (random.random() - 0.5) * 2.0
            data[i, spike_time:spike_time+5, 0] += spike_height
            logger.info(f"     (Injecting anomaly spike into sample {i})")

    logger.info(f"   -> Generated ECG data shape: {data.shape}")
    return data


def load_snn_model(
    model_config_path: str,
    model_path: Optional[str],
    device: str,
    num_classes: int,
    input_features: int
) -> Tuple[nn.Module, str]:
    """
    設定ファイルとオプションの重みファイルからSNNモデルをロードする。
    """
    if not Path(model_config_path).exists():
        logger.error(f"モデル設定ファイルが見つかりません: {model_config_path}")
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    cfg: DictConfig = cast(DictConfig, OmegaConf.load(model_config_path))

    # TSkipsSNN/SpikingSSM の設定を上書き
    architecture_type: str = cfg.model.get("architecture_type", "unknown")

    if architecture_type == "tskips_snn":
        OmegaConf.update(cfg, "model.input_features",
                         input_features, merge=True)
    elif architecture_type == "spiking_ssm":
        logger.warning(
            f"SpikingSSM ({architecture_type}) は現在テキスト入力前提です。ECGデモでは 'temporal_snn' または 'tskips_snn' を推奨します。")
        OmegaConf.update(cfg, "model.d_model", input_features, merge=True)

    # vocab_sizeをクラス数として渡す
    model_container = SNNCore(config=cfg.model, vocab_size=num_classes)
    # Cast to nn.Module to fix mypy error
    model: nn.Module = cast(nn.Module, model_container.model)  # SNNCore内部の実際のモデルを取得

    if model_path and Path(model_path).exists():
        logger.info(f"Loading trained model weights from: {model_path}")
        try:
            checkpoint: Dict[str, Any] = torch.load(
                model_path, map_location=device)
            state_dict: Dict[str, Any] = checkpoint.get(
                'model_state_dict', checkpoint)
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            logger.info("✅ Trained weights loaded successfully.")
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to load trained weights: {e}. Using initialized model.")
    else:
        # 修正: WARNING -> INFO に変更 (デモでは正常なフローのため)
        logger.info(
            "ℹ️ No trained model path provided or file not found. Using initialized model for demonstration.")

    model.to(device)
    model.eval()
    return model, architecture_type


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SNN ECG Anomaly Detection Demo")
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/models/ecg_temporal_snn.yaml",
        help="SNNモデル設定ファイル (例: ecg_temporal_snn.yaml, tskips_snn.yaml)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained SNN model weights (.pth)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of ECG samples to generate and classify."
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=500,
        help="Number of time steps for each ECG sample."
    )
    parser.add_argument(
        "--features",
        type=int,
        default=1,
        help="Number of input features (e.g., 1 for temporal_snn, 700 for SHD-based tskips_snn)"
    )
    args: argparse.Namespace = parser.parse_args()

    device: str = get_auto_device()
    logger.info(f"Using device: {device}")

    num_classes: int = 2  # 異常(1) / 正常(0)

    # 1. モデルのロード
    try:
        model, architecture_type = load_snn_model(
            args.model_config,
            args.model_path,
            device,
            num_classes,
            args.features
        )
        logger.info(
            f"Successfully loaded SNN model (Type: {architecture_type}).")
    except Exception as e:
        logger.error(f"モデルのロード中にエラーが発生しました: {e}")
        sys.exit(1)

    # 2. ダミーECGデータの生成
    ecg_data = generate_dummy_ecg_data(
        args.num_samples,
        args.time_steps,
        args.features,
        device
    )

    # 3. 推論の実行
    logger.info("\n--- Starting ECG Classification ---")
    results: List[Dict[str, Any]] = []

    with torch.no_grad():
        # モデルタイプに応じて入力キーを決定
        model_input: torch.Tensor
        input_key: str

        if architecture_type == "spiking_cnn":
            # (B, T, F) -> (B, F, T, 1) (F=C, T=H, 1=W)
            model_input = ecg_data.permute(0, 2, 1).unsqueeze(-1)
            input_key = "input_images"
            logger.info(
                f"   Input reshaped for SpikingCNN: {model_input.shape}")

        elif architecture_type in ["temporal_snn", "tskips_snn", "spiking_ssm", "gated_snn"]:
            model_input = ecg_data
            if architecture_type == "tskips_snn":
                input_key = "input_sequence"
            elif architecture_type == "spiking_ssm":
                input_key = "input_ids"
            elif architecture_type == "gated_snn":
                input_key = "input_sequence"
            else:
                input_key = "input_sequence"

            logger.info(
                f"   Input shape for {architecture_type}: {model_input.shape} (Key: {input_key})")

        else:
            logger.warning(
                f"   Input shape adjustment for model type '{architecture_type}' is not defined. Using raw shape.")
            model_input = ecg_data
            input_key = "input_ids"

        # SNNCoreラッパー経由で実行
        model_kwargs = {
            input_key: model_input,
            "return_spikes": True
        }

        try:
            # model は SNNCore.model (BaseModel)
            outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = model(
                **model_kwargs)  # type: ignore[operator]
            logits: torch.Tensor = outputs[0]
            avg_spikes_tensor: torch.Tensor = outputs[1]
            avg_spikes: float = avg_spikes_tensor.item(
            ) if avg_spikes_tensor is not None else 0.0
        except Exception as e:
            logger.error(
                f"推論中にエラーが発生しました (Input Key: {input_key}, Shape: {model_input.shape}): {e}", exc_info=True)
            sys.exit(1)

        if logits.dim() == 3:
            logger.info(
                "   Model returned sequence logits. Averaging over time...")
            logits = logits.mean(dim=1)

        if logits.shape[1] != num_classes:
            logger.error(
                f"モデルの出力次元 ({logits.shape[1]}) が期待されるクラス数 ({num_classes}) と一致しません。")
            sys.exit(1)

        predictions = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)

        for i in range(args.num_samples):
            pred_class: int = int(predictions[i].item())
            pred_label = "Anomaly" if pred_class == 1 else "Normal"
            confidence = probabilities[i, pred_class].item()
            results.append({
                "Sample": i + 1,
                "Prediction": pred_label,
                "Confidence": f"{confidence:.2%}",
            })

    # 4. 結果の表示
    print("\n--- Classification Results ---")
    if results:
        headers = list(results[0].keys())
        print(f"{' | '.join(headers)}")
        print("-" * (len(' | '.join(headers)) + 2))
        for row in results:
            print(f"{' | '.join(map(str, row.values()))}")
        print(f"\nAverage Spikes per Sample (Batch): {avg_spikes:.0f}")
    else:
        print("No results generated.")

    logger.info("\n✅ ECG analysis demo finished.")


if __name__ == "__main__":
    main()