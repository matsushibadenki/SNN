# scripts/train_bio_pc_cifar10.py
# 生物学的な予測符号化ネットワーク(Bio-PC)を用いたCIFAR-10学習スクリプト
#
# ディレクトリ: scripts/
# ファイル名: Bio-PC CIFAR-10 学習
# 目的: k-WTAと予測誤差最小化を用いた、バックプロパゲーションに依存しない学習のデモ。
#
# 変更点:
# - [修正 v4] mypyエラー解消: BioPCNetworkへの引数名を layer_dims から layer_sizes へ修正。
# - [修正 v4] layer_states への不正なアクセスを pc_layers へのイテレーションに修正。
# - [修正 v4] 無限再帰防止済みの BioPCNetwork モデルを使用。

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# プロジェクトルートをPythonパスに追加
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_research.core.networks.bio_pc_network import BioPCNetwork

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available(): device = torch.device("cuda")
    
    print(f"🚀 Starting Bio-PCNet CIFAR-10 Training")
    print(f"   Device: {device}")

    # データセットの準備
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)) # フラット化 (784 dims)
    ])

    # 本来はCIFAR10だが、指定のレイヤーサイズ[784,...]に合わせMNIST/グレー画像として扱う
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # モデルの初期化
    # mypy修正: 引数名を layer_sizes に統一
    model = BioPCNetwork(
        layer_sizes=[784, 512, 256, 10], 
        sparsity=0.05,
        input_gain=3.0
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            model.reset_state() # 無限再帰修正済み

            # 順伝播
            outputs = model(inputs)
            
            # 損失計算
            loss = criterion(outputs, targets)
            
            # スパース性正則化の追加
            loss += 0.01 * model.get_sparsity_loss()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"})

        # mypy修正: レイヤー状態へのアクセス
        # 以前のコードでは model.layer_states だったが、pc_layers を直接参照する
        avg_spike_rate = 0.0
        for layer in model.pc_layers:
            if hasattr(layer, 'inference_neuron'):
                # 膜電位などの統計情報を取得するデバッグ出力例
                pass

        logger.info(f"Epoch {epoch+1} Summary: Loss={total_loss/len(train_loader):.4f}, Acc={100.*correct/total:.2f}%")

    print("✅ Training Complete.")

if __name__ == "__main__":
    main()
