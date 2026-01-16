# ファイルパス: scripts/demos/visual/run_spiking_ff_demo.py
# 日本語タイトル: run_spiking_ff_demo
# 目的: データ拡張を削除し、学習率を戻して精度を回復させた安定版

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from snn_research.training.trainers.forward_forward import ForwardForwardTrainer

# Pickleエラー回避のため、Lambda関数ではなくモジュールレベルの関数として定義
def flatten_tensor(x):
    return torch.flatten(x)

def run_spiking_ff_mnist():
    print("=== Spiking Forward-Forward Experiment (True SNN) - Stable Tuning ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # 1. データセット設定
    batch_size = 128
    
    # 【修正】データ拡張（RandomAffine）を削除し、テスト用と同じ設定に戻す
    # 全結合層(Linear)は位置ずれに弱いため、Augmentationは逆効果でした
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten_tensor)
    ])
    
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    # train, test共に同じtransformを使用
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
    # 成功実績のあるサイズ設定
    print("Initializing Spiking Neural Network...")
    model = nn.Sequential(
        nn.Linear(784, 1500),
        nn.Linear(1500, 1500),
    )

    # 3. Trainer設定
    # 学習率を安定版(0.0015)に戻す
    config = {
        "use_snn": True,
        "time_steps": 25,
        "learning_rate": 0.0015,  # ← 0.002から戻しました
        "ff_threshold": 0.6,      # 成功した閾値を維持
        "num_epochs": 30,         # 長めのエポック数は維持
        "snn_tau": 0.5,
        "snn_threshold": 1.0,
        "snn_reset": "subtract"
    }

    print(f"Configuration: Threshold={config['ff_threshold']}, LR={config['learning_rate']}")
    print("Setting up SNN Trainer...")
    trainer = ForwardForwardTrainer(
        model=model,
        device=device,
        config=config,
        num_classes=10
    )

    # 4. 学習率スケジューラの設定
    # 後半の微調整用として残す
    schedulers = []
    if hasattr(trainer, 'ff_layers'):
        for layer in trainer.ff_layers:
            schedulers.append(CosineAnnealingLR(layer.optimizer, T_max=config['num_epochs']))
    else:
        print("Warning: Could not setup schedulers (ff_layers not found)")

    # 5. 学習ループ
    print(f"Start SNN Training for {config['num_epochs']} epochs with Scheduler...")
    
    for epoch in range(1, config['num_epochs'] + 1):
        metrics = trainer.train_epoch(train_loader, epoch)
        
        # スケジューラの更新
        current_lr = 0.0
        for scheduler in schedulers:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        # ログ出力
        print(f"Epoch {epoch} [LR: {current_lr:.6f}] - Loss: {metrics['train_loss']:.4f} "
              f"| Pos: {metrics['avg_pos_goodness']:.4f} "
              f"| Neg: {metrics['avg_neg_goodness']:.4f}")

        # 検証
        if epoch % 1 == 0:
            # print(f"Validating Epoch {epoch}...") # ログが見づらくなるのでコメントアウト推奨ですが、進捗確認のため残します
            acc = trainer.predict(test_loader)
            print(f"Epoch {epoch} - SNN Accuracy: {acc:.2f}%")

    print("Experiment Finished.")
    
    save_dir = os.path.join(os.path.dirname(__file__), '../../results/checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_dir, "spiking_ff_model_stable.pth"))

if __name__ == "__main__":
    run_spiking_ff_mnist()