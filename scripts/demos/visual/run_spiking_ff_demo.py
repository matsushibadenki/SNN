# ファイルパス: scripts/demos/visual/run_spiking_ff_demo.py
# 日本語タイトル: run_spiking_ff_demo
# 目的: SNN Forward-Forward学習の精度向上版（学習率スケジューラ・データ拡張導入）

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
    print("=== Spiking Forward-Forward Experiment (True SNN) - Advanced Tuning ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # 1. データセットとデータ拡張
    batch_size = 128
    
    # 学習用: ランダムな変形を加えて汎化性能を上げる
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten_tensor)
    ])
    
    # テスト用: 変形なし
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten_tensor)
    ])

    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=test_transform)

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
    # 層のサイズを維持（前回良い結果が出たため）
    print("Initializing Spiking Neural Network...")
    model = nn.Sequential(
        nn.Linear(784, 1500),
        nn.Linear(1500, 1500),
    )

    # 3. Trainer設定
    # エポック数を増やし、学習率スケジューラで最後は微調整を行う
    config = {
        "use_snn": True,
        "time_steps": 25,
        "learning_rate": 0.002,   # 初期学習率は少し高めに設定
        "ff_threshold": 0.6,      # 成功した閾値を維持
        "num_epochs": 30,         # スケジューラ効果を出すためエポック延長
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
    # ForwardForwardTrainerは層ごとにオプティマイザを持っているので、個別にスケジューラを設定
    schedulers = []
    if hasattr(trainer, 'ff_layers'):
        for layer in trainer.ff_layers:
            # Cosine Annealing: 学習率を徐々に0付近まで下げる
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
            current_lr = scheduler.get_last_lr()[0] # ログ用

        # ログ出力
        print(f"Epoch {epoch} [LR: {current_lr:.6f}] - Loss: {metrics['train_loss']:.4f} "
              f"| Pos: {metrics['avg_pos_goodness']:.4f} "
              f"| Neg: {metrics['avg_neg_goodness']:.4f}")

        # 数エポックごとに評価（頻繁すぎると時間がかかるため）
        if epoch % 1 == 0:  # 毎回確認したい場合は1のまま
            print(f"Validating Epoch {epoch}...")
            acc = trainer.predict(test_loader)
            print(f"Epoch {epoch} - SNN Accuracy: {acc:.2f}%")

    print("Experiment Finished.")
    
    save_dir = os.path.join(os.path.dirname(__file__), '../../results/checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_dir, "spiking_ff_model_tuned.pth"))

if __name__ == "__main__":
    run_spiking_ff_mnist()