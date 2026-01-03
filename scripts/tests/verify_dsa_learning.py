# ファイルパス: scripts/verify_dsa_learning.py
# 日本語タイトル: DSA学習能力検証スクリプト
# 目的・内容:
#   実装した DSASpikingTransformer が実際に学習（Lossの低下と精度の向上）
#   可能であることを検証するための軽量な学習ループ。
#   データセット: ランダムなベクトル列の分類（ダミーデータ）

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.models.transformer.dsa_transformer import DSASpikingTransformer

def generate_dummy_data(num_samples=200, seq_len=20, input_dim=16, num_classes=2):
    """
    ダミーデータの生成。
    クラス0: 前半の値が大きい傾向
    クラス1: 後半の値が大きい傾向
    という単純な時間的特徴を持たせる。
    """
    X = torch.randn(num_samples, seq_len, input_dim)
    Y = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        if torch.rand(1).item() > 0.5:
            # Class 1: Add bias to second half
            X[i, seq_len//2:, :] += 1.0
            Y[i] = 1
        else:
            # Class 0: Add bias to first half
            X[i, :seq_len//2, :] += 1.0
            Y[i] = 0
            
    return X, Y

def main():
    print("🚀 Starting SNN-DSA Learning Verification...")
    
    # 設定
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 10 # 動作確認用なので短く
    LR = 1e-3
    
    # モデルパラメータ (Phase 8-2: Sparse Attention)
    model_config = {
        'input_dim': 16,
        'output_dim': 2,
        'd_model': 32,
        'num_heads': 4,
        'num_layers': 2,
        'top_k': 5, # 時間方向(20ステップ)に対してTop-5のみ注目
        'max_len': 20
    }
    
    # データ準備
    X, Y = generate_dummy_data(num_samples=200, seq_len=20, input_dim=16)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # モデル初期化
    model = DSASpikingTransformer(**model_config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model initialized on {DEVICE}. Config: {model_config}")
    print(f"Dataset: {len(dataset)} samples.")
    
    # 学習ループ
    loss_history = []
    acc_history = []
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # SNN状態リセット
            model.reset_state()
            
            # Forward
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング (SNN学習安定化のため)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
    # 検証結果の判定
    final_acc = acc_history[-1]
    print("\n📊 Verification Result:")
    if final_acc > 80.0:
        print(f"✅ PASSED: Model achieved {final_acc:.2f}% accuracy (Target > 80%).")
        print("   DSA mechanism successfully propagated gradients and learned temporal patterns.")
    else:
        print(f"⚠️  WARNING: Final accuracy {final_acc:.2f}% is low. Parameter tuning might be needed.")

    # 可視化画像を保存 (オプション)
    try:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(acc_history, label='Accuracy', color='orange')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        os.makedirs("results", exist_ok=True)
        plt.savefig("workspace/results/dsa_verification_plot.png")
        print("   Plot saved to results/dsa_verification_plot.png")
    except Exception as e:
        print(f"   (Plotting skipped: {e})")

if __name__ == "__main__":
    main()