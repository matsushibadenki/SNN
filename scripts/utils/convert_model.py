# scripts/convert_model.py
# ANNモデルからSNNモデルへの変換・蒸留を実行するためのスクリプト
#
# ディレクトリ: scripts/convert_model.py
# ファイル名: ANN-SNN 高忠実度変換スクリプト
# 目的: 学習済みANN（CNN/LLM）の重みを抽出し、SNNアーキテクチャへ適合・変換する。
#
# 変更点:
# - [修正 v8] チェックポイントファイルが辞書形式(epoch, model_state_dict等を含む)の場合に対応。
# - [修正 v8] model_state_dict 抽出後に DataParallel の 'module.' プレフィックス除去を行うよう順序を調整。
# - [修正 v7] print(..., flush=True) を使用して出力を即座にフラッシュ。
# - [修正 v7] ロガー設定を main 関数の先頭で再設定。

from snn_research.benchmarks.ann_baseline import SimpleCNN
from snn_research.conversion.ann_to_snn_converter import AnnToSnnConverter
from app.containers import TrainingContainer
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))


def get_calibration_loader(container):
    """キャリブレーション用の小規模なデータローダーを返す"""
    # 既存の機能を維持しつつ安全に取得
    try:
        vocab_size = container.tokenizer.provided.vocab_size()
    except Exception:
        vocab_size = 32000  # デフォルト
    dummy_data = torch.randint(0, vocab_size, (128, 32))  # 多様なサンプル
    dummy_dataset = TensorDataset(dummy_data)
    return DataLoader(dummy_dataset, batch_size=16)


def main():
    # --- ロギングの強制再設定 ---
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="ANNモデルからSNNへの高忠実度変換ツール",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 共通引数
    parser.add_argument("--ann_model_path", type=str, required=True,
                        help="変換元の学習済みANNモデルのパスまたはHugging Face ID。")
    parser.add_argument("--snn_model_config", type=str,
                        required=True, help="変換先のSNNモデルのアーキテクチャ設定ファイル。")
    parser.add_argument("--output_snn_path", type=str,
                        required=True, help="変換後にSNNモデルを保存するパス。")
    parser.add_argument("--method", type=str, required=True,
                        choices=["cnn-convert", "llm-convert"], help="実行する変換メソッド。")

    # オプション引数
    parser.add_argument("--dry-run", action="store_true",
                        help="実際の変換処理を実行せず、設定とファイルのチェックのみ行う。")

    args = parser.parse_args()

    # DIコンテナからSNNモデルのインスタンスと設定を取得
    try:
        container = TrainingContainer()
        container.config.from_yaml("configs/templates/base_config.yaml")
        container.config.from_yaml(args.snn_model_config)
        snn_model = container.snn_model()
        snn_config = container.config.model.to_dict()
    except Exception as e:
        logger.error(f"設定ファイルの読み込みまたはモデルの初期化に失敗しました: {e}")
        sys.exit(1)

    msg = "✅ SNNモデルと設定の準備が完了しました。"
    logger.info(msg)
    print(msg, flush=True)

    if args.dry_run:
        logger.info("--dry-run モード: 実際の変換は行わずに終了します。")
        print("--dry-run モード: 実際の変換は行わずに終了します。", flush=True)
        sys.exit(0)

    converter = AnnToSnnConverter(snn_model=snn_model, model_config=snn_config)
    calibration_loader = get_calibration_loader(container)

    try:
        if args.method == "cnn-convert":
            logger.info(
                f"CNN変換を開始します: {args.ann_model_path} -> {args.output_snn_path}")
            ann_model = SimpleCNN(num_classes=10)
            checkpoint = torch.load(args.ann_model_path, map_location='cpu')

            # チェックポイントが辞書形式（model_state_dictを含む）か、state_dict単体かを判定
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # DataParallel ('module.') のプレフィックス除去
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', '', 1)                              : v for k, v in state_dict.items()}

            ann_model.load_state_dict(state_dict)
            converter.convert_cnn_weights(
                ann_model, args.output_snn_path, calibration_loader)

        elif args.method == "llm-convert":
            logger.info(
                f"LLM変換を開始します: {args.ann_model_path} -> {args.output_snn_path}")
            converter.convert_llm_weights(
                ann_model_name_or_path=args.ann_model_path,
                output_path=args.output_snn_path,
                calibration_loader=calibration_loader
            )

        logger.info(f"✅ 変換が成功しました。保存先: {args.output_snn_path}")
        print(f"✅ 変換が成功しました。保存先: {args.output_snn_path}", flush=True)

    except Exception as e:
        logger.error(f"変換プロセス中に致命的なエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
