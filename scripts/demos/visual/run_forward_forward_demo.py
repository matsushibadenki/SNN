# ファイルパス: scripts/runners/run_forward_forward_demo.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

# プロジェクトルートにパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# トレーナーのインポート
from snn_research.training.trainers.forward_forward import ForwardForwardTrainer

def run_forward_forward_mnist():
    """
    Forward-Forwardアルゴリズムを用いてMNISTを学習させるデモ実験。
    バックプロパゲーションを使わずに数字分類を学習します。
    """
    print("=== Forward-Forward Algorithm Experiment (MNIST) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # データセットの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), 
        transforms.Lambda(lambda x: torch.flatten(x)) 
    ])

    print("Downloading/Loading MNIST dataset...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # モデルの定義
    model = nn.Sequential(
        nn.Linear(784, 500),
        nn.ReLU(),
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, 500), 
        nn.ReLU()
    )

    # トレーナーの設定
    # 学習率: 0.001 (Mean Goodness, Adam用)
    # Threshold: 2.0 (Mean Goodnessの閾値として標準的)
    config = {
        "learning_rate": 0.001,
        "ff_threshold": 2.0, 
        "num_epochs": 20
    }

    trainer = ForwardForwardTrainer(
        model=model,
        device=device,
        config=config,
        num_classes=10
    )

    # 学習ループ
    print(f"Start training for {config['num_epochs']} epochs...")
    
    for epoch in range(1, config['num_epochs'] + 1):
        metrics = trainer.train_epoch(train_loader, epoch)
        
        print(f"Validating Epoch {epoch}...")
        acc = trainer.predict(test_loader)
        
        print(f"Epoch {epoch}/{config['num_epochs']} - Loss: {metrics['train_loss']:.4f} - Test Accuracy: {acc:.2f}%")

    print("Experiment finished.")
    
    # モデルの保存
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ff_mnist_model.pth")
    trainer.save_checkpoint(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    run_forward_forward_mnist()