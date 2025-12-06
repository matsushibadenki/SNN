# scripts/convert_model.py
# (更新: ログ出力の強制)
# ANNモデルからSNNモデルへの変換・蒸留を実行するためのスクリプト
#
# 変更点:
# - [修正 v6] logging.basicConfig に force=True を追加。
# - [修正 v6] 完了メッセージを print() でも出力するように変更。

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import TrainingContainer
from snn_research.conversion.ann_to_snn_converter import AnnToSnnConverter
from snn_research.benchmark.ann_baseline import SimpleCNN
from omegaconf import OmegaConf

# --- 修正: ストリームを標準出力に設定し、設定を強制 ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True # 他のライブラリによる設定を上書き
)

def get_calibration_loader(container):
    """キャリブレーション用の小規模なデータローダーを返す"""
    vocab_size = container.tokenizer.provided.vocab_size()
    dummy_data = torch.randint(0, vocab_size, (128, 32)) # 多様なサンプル
    dummy_dataset = TensorDataset(dummy_data)
    return DataLoader(dummy_dataset, batch_size=16)

def main():
    parser = argparse.ArgumentParser(
        description="ANNモデルからSNNへの高忠実度変換ツール",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 共通引数
    parser.add_argument("--ann_model_path", type=str, required=True, help="変換元の学習済みANNモデルのパスまたはHugging Face ID。")
    parser.add_argument("--snn_model_config", type=str, required=True, help="変換先のSNNモデルのアーキテクチャ設定ファイル。")
    parser.add_argument("--output_snn_path", type=str, required=True, help="変換後にSNNモデルを保存するパス。")
    parser.add_argument("--method", type=str, required=True, choices=["cnn-convert", "llm-convert"], help="実行する変換メソッド。")
    
    # オプション引数
    parser.add_argument("--dry-run", action="store_true", help="実際の変換処理を実行せず、設定とファイルのチェックのみ行う。")
    
    args = parser.parse_args()

    # DIコンテナからSNNモデルのインスタンスと設定を取得
    try:
        container = TrainingContainer()
        container.config.from_yaml("configs/templates/base_config.yaml")
        container.config.from_yaml(args.snn_model_config)
        snn_model = container.snn_model()
        snn_config = container.config.model.to_dict()
    except Exception as e:
        logging.error(f"設定ファイルの読み込みまたはモデルの初期化に失敗しました: {e}")
        sys.exit(1)

    # --- 修正: printでも出力 ---
    msg = "✅ SNNモデルと設定の準備が完了しました。"
    logging.info(msg)
    print(msg) 

    if args.dry_run:
        logging.info("--dry-run モード: 実際の変換は行わずに終了します。")
        sys.exit(0)
        
    converter = AnnToSnnConverter(snn_model=snn_model, model_config=snn_config)
    calibration_loader = get_calibration_loader(container)

    try:
        if args.method == "cnn-convert":
            logging.info(f"CNN変換を開始します: {args.ann_model_path} -> {args.output_snn_path}")
            ann_model = SimpleCNN(num_classes=10) 
            state_dict = torch.load(args.ann_model_path, map_location='cpu')
            if list(state_dict.keys())[0].startswith('module.'):
                 state_dict = {k[7:]: v for k, v in state_dict.items()}
            ann_model.load_state_dict(state_dict)
            converter.convert_cnn_weights(ann_model, args.output_snn_path, calibration_loader)

        elif args.method == "llm-convert":
            logging.info(f"LLM変換を開始します: {args.ann_model_path} -> {args.output_snn_path}")
            converter.convert_llm_weights(
                ann_model_name_or_path=args.ann_model_path,
                output_path=args.output_snn_path,
                calibration_loader=calibration_loader
            )
    except Exception as e:
        logging.error(f"変換プロセス中に致命的なエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
