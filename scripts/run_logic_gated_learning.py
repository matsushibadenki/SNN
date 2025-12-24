# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Final: Surrogate Gradient版)

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 5000, in_features: int = 784, out_features: int = 10):
    # バイナリ入力
    x = (torch.randn(num_samples, in_features) > 0.0).float()
    y = []
    for i in range(num_samples):
        # ターゲット領域: 200-260 (60ビット)
        sum_val = x[i, 200:260].sum().long() 
        val = sum_val % out_features
        y.append(val)
    
    y = torch.stack(y)
    # CrossEntropyLossはクラスインデックスを受け取るのでOne-hotにはしない
    return x, y

def run_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    # Hidden層: 2048 (十分な容量)
    core = HybridNeuromorphicCore(784, 2048, 10).to(device)
    
    total_samples = 20000
    batch_size = 128 # 安定した勾配計算のために大きめ
    
    print("\nGenerating Data...")
    x_train, y_train = generate_synthetic_data(num_samples=total_samples)
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Surrogate Gradient Mode)...")
    
    ma_acc = 0.1
    epochs = 50
    
    for epoch in range(epochs):
        core.train()
        epoch_correct = 0
        total_seen = 0
        
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            
            batch_acc = metrics["reward"]
            ma_acc = ma_acc * 0.95 + batch_acc * 0.05
            
            epoch_correct += batch_acc * data.size(0)
            total_seen += data.size(0)
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1:2d} [{i*batch_size:5d}/{total_samples}] - "
                      f"Loss: {metrics['prediction_error']:.4f} | "
                      f"Acc(Batch): {batch_acc*100:.1f}% | "
                      f"Acc(MA): {ma_acc*100:.1f}% | "
                      f"Spikes: {metrics['output_spike_count']:.1f}")
        
        epoch_acc = epoch_correct / total_seen * 100
        print(f"--- Epoch {epoch+1} Final Accuracy: {epoch_acc:.2f}% ---")
        
        if epoch_acc > 95.0:
            print("Target Accuracy Reached. Early Stopping.")
            break

    print("\nRunning Final Evaluation...")
    core.eval()
    test_samples = 2000
    x_test, y_test = generate_synthetic_data(num_samples=test_samples)
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    correct_test = 0
    with torch.no_grad():
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        for data, target in test_loader:
            out = core(data)
            pred = out.argmax(dim=1)
            correct_test += (pred == target).float().sum().item()
                
    print(f"Final Test Accuracy: {correct_test/test_samples*100:.2f}%")

if __name__ == "__main__":
    run_simulation()
