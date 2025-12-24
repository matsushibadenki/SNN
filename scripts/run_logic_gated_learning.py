# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Final: 高精度達成版)

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 5000, in_features: int = 784, out_features: int = 10):
    # ノイズを減らし、明確なシグナルにする
    # 平均0, 分散1の正規分布 -> 0.0より大きければ1 (約50%の密度)
    x = (torch.randn(num_samples, in_features) > 0.0).float()
    y = []
    for i in range(num_samples):
        # ターゲット領域: 200-260 (60ビット)
        sum_val = x[i, 200:260].sum().long() 
        val = sum_val % out_features
        y.append(val)
    
    y = torch.stack(y)
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    # Hidden層: 2048 (容量を倍増して、表現力を確保)
    core = HybridNeuromorphicCore(784, 2048, 10).to(device)
    
    total_samples = 20000 # 学習データ量を倍増
    # バッチサイズを64にして安定化
    batch_size = 64
    
    print("\nGenerating Data...")
    x_train, y_train = generate_synthetic_data(num_samples=total_samples)
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (High-Density Init Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    epochs = 40 # じっくり学習
    
    for epoch in range(epochs):
        epoch_correct = 0
        total_seen = 0
        
        core.train()
        
        for i, (data, target_onehot) in enumerate(loader):
            target_idx = target_onehot.argmax(dim=1)
            metrics = core.autonomous_step(data, target_idx)
            
            with torch.no_grad():
                out = core(data)
                pred = out.argmax(dim=1)
                correct = (pred == target_idx).float().sum().item()
                epoch_correct += correct
                total_seen += data.size(0)
            
            batch_acc = correct / data.size(0)
            correct_avg = correct_avg * 0.95 + batch_acc * 0.05
            
            if i % 100 == 0:
                w_hid = core.fast_process.get_ternary_weights()
                conn_hid = float(w_hid.mean().item()) * 100
                w_out = core.output_gate.get_ternary_weights()
                conn_out = float(w_out.mean().item()) * 100
                out_spikes = metrics["output_spike_count"]
                
                print(f"Epoch {epoch+1:2d} [{i*batch_size:5d}/{total_samples}] - "
                      f"Acc(Batch): {batch_acc*100:.1f}% | "
                      f"Acc(MA): {correct_avg*100:.1f}% | "
                      f"Conn(H): {conn_hid:.1f}% | "
                      f"Spikes: {out_spikes:.1f}")
        
        epoch_acc = epoch_correct / total_seen * 100
        print(f"--- Epoch {epoch+1} Final Accuracy: {epoch_acc:.2f}% ---")
        
        if epoch_acc > 92.0:
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
        
        for data, target_onehot in test_loader:
            target = target_onehot.argmax(dim=1)
            out = core(data)
            pred = out.argmax(dim=1)
            correct_test += (pred == target).float().sum().item()
                
    print(f"Final Test Accuracy: {correct_test/test_samples*100:.2f}%")

if __name__ == "__main__":
    run_simulation()
