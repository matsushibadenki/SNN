# ファイルパス: scripts/training/train_phase2.py
# 日本語タイトル: Phase 2 Advanced Training Script (MPS/tqdm対応版)
# 目的: 1.58bit Spikformerモデルを用いて、高精度かつ堅牢なモデルを学習する。
#       進捗表示とMac M-series (MPS) 加速を追加。

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm  # プログレスバー用
import sys

# プロジェクトルートの設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from snn_research.models.transformer.spikformer import Spikformer
from snn_research.metrics.energy import EnergyMeter

                            
def train_phase2():
    # 設定
    BATCH_SIZE = 64
    EPOCHS = 50 
    LR = 1e-3
    
    # デバイス設定: Mac (MPS) > NVIDIA (CUDA) > CPU
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("🚀 Using CUDA GPU")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("🚀 Using Apple MPS (M-series GPU)")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️ Using CPU (Slow training expected)")
    
    print(f"   Target: Accuracy > 97.5%, Energy Efficiency 100x")

    # データセット (CIFAR-10) with Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Mac (MPS) 用の最適化設定
    if DEVICE.type == 'mps':
        num_workers = 0  # MPSでは0の方が速いことが多い
        pin_memory = False # MPSではTrueにすると不安定な場合があるためFalse推奨
        print("⚡ Optimized for Apple MPS: num_workers=0")
    else:
        num_workers = 4
        pin_memory = True

    # ダウンロード
    trainset = datasets.CIFAR10(root='./workspace/data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=num_workers, pin_memory=pin_memory)
    
    testset = datasets.CIFAR10(root='./workspace/data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)

    # モデル初期化
    model = Spikformer(
        img_size_h=32, img_size_w=32,
        patch_size=4, in_channels=3,
        embed_dim=256, num_heads=8, num_layers=4,
        T=4, num_classes=10
    ).to(DEVICE)
    
    # オプティマイザ
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    
    # スケジューラ
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 損失関数
    criterion = nn.CrossEntropyLoss()

    # エネルギーメーター
    energy_meter = EnergyMeter()

    # 学習ループ
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # tqdmでプログレスバーを表示
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 統計更新
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # プログレスバーに情報を表示
            current_acc = 100. * correct / total
            pbar.set_postfix({'Loss': f"{running_loss/(i+1):.4f}", 'Acc': f"{current_acc:.2f}%"})

        train_acc = 100. * correct / total
        
        # 評価
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        scheduler.step()
        
        # エポック終了後のログ
        print(f"Epoch [{epoch+1}/{EPOCHS}] Final -> "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # チェックポイント保存
        if test_acc > 95.0:
            save_path = os.path.join(project_root, "workspace", "models", f"spikformer_phase2_epoch{epoch}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  💾 High Accuracy Model Saved: {save_path}")

    print("🏁 Training Finished.")

if __name__ == "__main__":
    train_phase2()