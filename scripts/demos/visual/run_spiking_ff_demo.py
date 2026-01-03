# ファイルパス: scripts/runners/run_spiking_ff_demo.py
# 日本語タイトル: run_spiking_ff_demo
# 目的: 

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from snn_research.training.trainers.forward_forward import ForwardForwardTrainer

def run_spiking_ff_mnist():
    print("=== Spiking Forward-Forward Experiment (True SNN) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # 1. データセット
    # SNNの場合、ピクセル値を電流として扱うため、0-1正規化や適度なスケーリングが重要
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x)) # 全結合SNNのためFlatten
    ])

    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 2. モデル定義
    # ReLUは不要です（Trainerが自動的にLIFに置き換えるか、無視してSNN層を作ります）
    # ここでは構造だけ定義します。
    print("Initializing Spiking Neural Network...")
    model = nn.Sequential(
        nn.Linear(784, 1000),
        nn.Linear(1000, 1000),
        nn.Linear(1000, 1000)
    )

    # 3. Trainer設定
    # use_snn=True にすることで、SpikingForwardForwardLayerが使用されます
    config = {
        "use_snn": True,          # SNNモード有効化
        "time_steps": 20,         # シミュレーション時間ステップ数
        "learning_rate": 0.005,   # SNNは学習率低めが良い
        "ff_threshold": 0.08,     # 発火率の閾値 (0.0~1.0の間。疎な発火を目指すなら低く)
        "num_epochs": 10
    }

    print("Setting up SNN Trainer...")
    trainer = ForwardForwardTrainer(
        model=model,
        device=device,
        config=config,
        num_classes=10
    )

    # 4. 学習ループ
    print(f"Start SNN Training for {config['num_epochs']} epochs...")
    
    for epoch in range(1, config['num_epochs'] + 1):
        metrics = trainer.train_epoch(train_loader, epoch)
        
        # 評価
        print(f"Validating Epoch {epoch}...")
        acc = trainer.predict(test_loader)
        
        print(f"Epoch {epoch} - Loss: {metrics['train_loss']:.4f} - SNN Accuracy: {acc:.2f}%")

    print("Experiment Finished.")
    
    save_dir = os.path.join(os.path.dirname(__file__), '../../results/checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_dir, "spiking_ff_model.pth"))

if __name__ == "__main__":
    run_spiking_ff_mnist()