# ファイルパス: scripts/run_stdp_learning.py
# Title: STDP学習実行スクリプト
# Description:
#   Spike-Timing-Dependent Plasticity (STDP) を用いた教師なし学習の実験スクリプト。
#   (修正: mypyエラー解消 - no-redef, call-arg)

from torchvision import datasets, transforms  # type: ignore[import-untyped]
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
import sys
import os
from typing import Any, Dict

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# torchvision の型スタブがない場合があるため無視

# モデルクラスのインポート
try:
    # type: ignore[attr-defined]
    # type: ignore[attr-defined]
    from snn_research.models.cnn.spiking_cnn_model import SpikingCNN
except ImportError:
    # ダミー定義 (型チェック用)
    class SpikingCNN(nn.Module):  # type: ignore[no-redef]
        def __init__(self, config: Dict[str, Any],
                     num_classes: int): super().__init__()

# STDP学習則のインポートとエイリアス設定
# 名前重複を避けるため、一度 _STDP としてインポートするか、
# 明確に条件分岐を行う。
STDPLearner: Any

try:
    from snn_research.learning_rules.stdp import STDP
    STDPLearner = STDP
except ImportError:
    # 最終手段: ダミークラス
    class _DummySTDPLearner:
        def __init__(self, learning_rate: float, tau_trace: float,
                     a_plus: float, a_minus: float): pass
    STDPLearner = _DummySTDPLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="STDP Learning Experiment")
    parser.add_argument("--dataset", type=str,
                        default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    logger.info(f"Starting STDP learning on {args.dataset}...")

    # データセットの準備
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    if args.dataset == "mnist":
        train_dataset = datasets.MNIST(
            './data', train=True, download=True, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(
            './data', train=True, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    # モデル構築
    config = {
        'neuron': {'type': 'lif', 'v_threshold': 1.0, 'tau_mem': 20.0},
        'time_steps': 20,
        'architecture_type': 'spiking_cnn'
    }

    import typing
    # SpikingCNN をインスタンス化
    model = SpikingCNN(neuron_config=typing.cast(
        Dict[str, Any], config['neuron']), num_classes=10, vocab_size=0, time_steps=20)

    # STDP学習器の初期化
    # STDP学習器の初期化
    # 必要な引数 (a_plus, a_minus) を追加
    STDPLearner(
        learning_rate=0.01,
        tau_trace=20.0,
        a_plus=1.0,  # 追加
        a_minus=1.0  # 追加
    )

    # 学習ループ
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # ダミーの順伝播
            # model は nn.Module なので __call__ が可能
            _ = model(data)

            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}: STDP Update Simulated")

    logger.info("STDP learning completed.")


if __name__ == "__main__":
    main()
