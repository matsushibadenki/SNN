# ファイルパス: scripts/demos/visual/run_spiking_ff_demo.py
# 日本語タイトル: run_spiking_ff_demo
# 目的: 閾値を修正し、SNNの学習を正常化する

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from snn_research.training.trainers.forward_forward import ForwardForwardTrainer

def flatten_tensor(x):
    return torch.flatten(x)

def run_spiking_ff_mnist():
    print("=== Spiking Forward-Forward Experiment (True SNN) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # 1. データセット
    batch_size = 128  # バッチサイズ調整
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # SNNへの入力電流として適切な強さに調整（少し強めに）
        transforms.Normalize((0.1307,), (0.3081,)), 
        transforms.Lambda(flatten_tensor)
    ])

    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    use_pin_memory = True if device == "cuda" else False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1000, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=use_pin_memory
    )

    # 2. モデル定義
    # 学習しやすいサイズに設定
    print("Initializing Spiking Neural Network...")
    model = nn.Sequential(
        nn.Linear(784, 1500),
        nn.Linear(1500, 1500),
    )

    # 3. Trainer設定
    # 【重要】閾値をGoodnessのスケール(Mean)に合わせて小さく設定
    config = {
        "use_snn": True,
        "time_steps": 25,         # 時間方向の情報量を確保
        "learning_rate": 0.0015,  # 学習率
        "ff_threshold": 0.6,      # ← 【修正】平均二乗誤差ベースなので0.5〜1.0程度が適切
        "num_epochs": 15,
        "snn_tau": 0.5,           # 膜時定数
        "snn_threshold": 1.0,     # スパイク発火閾値
        "snn_reset": "subtract"   # Soft Reset
    }

    print(f"Configuration: Threshold={config['ff_threshold']}, LR={config['learning_rate']}")
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
        
        # ログ出力（Goodnessの状態を確認）
        print(f"Epoch {epoch} - Loss: {metrics['train_loss']:.4f} "
              f"| Goodness Pos: {metrics['avg_pos_goodness']:.4f} "
              f"| Goodness Neg: {metrics['avg_neg_goodness']:.4f}")

        # 評価
        print(f"Validating Epoch {epoch}...")
        acc = trainer.predict(test_loader)
        print(f"Epoch {epoch} - SNN Accuracy: {acc:.2f}%")

    print("Experiment Finished.")
    
    save_dir = os.path.join(os.path.dirname(__file__), '../../results/checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_dir, "spiking_ff_model_fixed.pth"))

if __name__ == "__main__":
    run_spiking_ff_mnist()