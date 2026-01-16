# ファイルパス: scripts/demos/visual/run_spiking_ff_demo.py
# 日本語タイトル: run_spiking_ff_demo
# 目的: 改善されたSNN-FFロジックを用いて高精度を目指す実行スクリプト

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from snn_research.training.trainers.forward_forward import ForwardForwardTrainer

def flatten_tensor(x):
    return torch.flatten(x)

def run_spiking_ff_mnist():
    print("=== Spiking Forward-Forward Experiment (True SNN) - Performance Tuned ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # 1. データセット設定
    batch_size = 128
    
    # 複雑なAugmentationは避け、正規化のみとする（Hard NegativeがAugmentationの役割を兼ねるため）
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten_tensor)
    ])
    
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=base_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=base_transform)

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
    # 層サイズは維持
    print("Initializing Spiking Neural Network...")
    model = nn.Sequential(
        nn.Linear(784, 1500),
        nn.Linear(1500, 1500),
    )

    # 3. Trainer設定
    # Adaptive Threshold導入により ff_threshold は「初期値」としての意味になる
    config = {
        "use_snn": True,
        "time_steps": 25,
        "learning_rate": 0.0015,
        "ff_threshold": 2.0,      # 初期値。学習が進むと自動的に適切な値(Pos mean)に収束する
        "num_epochs": 30,
        "snn_tau": 0.5,
        "snn_threshold": 1.0,
        "snn_reset": "subtract"
    }

    print(f"Configuration: Adaptive Threshold Init={config['ff_threshold']}, LR={config['learning_rate']}")
    print("Setting up SNN Trainer with V_mem Goodness & Hard Negatives...")
    trainer = ForwardForwardTrainer(
        model=model,
        device=device,
        config=config,
        num_classes=10
    )

    # 4. 学習率スケジューラの設定
    schedulers = []
    if hasattr(trainer, 'ff_layers'):
        for layer in trainer.ff_layers:
            schedulers.append(CosineAnnealingLR(layer.optimizer, T_max=config['num_epochs']))
    else:
        print("Warning: Could not setup schedulers (ff_layers not found)")

    # 5. 学習ループ
    print(f"Start SNN Training for {config['num_epochs']} epochs...")
    
    for epoch in range(1, config['num_epochs'] + 1):
        metrics = trainer.train_epoch(train_loader, epoch)
        
        # スケジューラの更新
        current_lr = 0.0
        for scheduler in schedulers:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        # ログ出力
        # Adaptive Thresholdの現在値をログに出せるとベストだが、レイヤーごとなので平均Goodnessで代替確認
        print(f"Epoch {epoch} [LR: {current_lr:.6f}] - Loss: {metrics['train_loss']:.4f} "
              f"| Pos(Energy): {metrics['avg_pos_goodness']:.4f} "
              f"| Neg(Energy): {metrics['avg_neg_goodness']:.4f}")

        # 検証 (頻度は適宜調整)
        if epoch % 1 == 0:
            acc = trainer.predict(test_loader)
            print(f"Epoch {epoch} - SNN Accuracy: {acc:.2f}%")

    print("Experiment Finished.")
    
    save_dir = os.path.join(os.path.dirname(__file__), '../../results/checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_dir, "spiking_ff_model_optimized.pth"))

if __name__ == "__main__":
    run_spiking_ff_mnist()