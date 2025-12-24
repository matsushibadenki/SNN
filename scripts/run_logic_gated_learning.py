# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Final Fix: 分離学習版)

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 5000, in_features: int = 784, out_features: int = 10):
    x = (torch.randn(num_samples, in_features) > 1.0).float()
    y = []
    for i in range(num_samples):
        sum_val = x[i, 200:260].sum().long() 
        val = sum_val % out_features
        y.append(val)
    
    y = torch.stack(y)
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    # Hidden層: 1024
    core = HybridNeuromorphicCore(784, 1024, 10).to(device)
    
    total_samples = 5000
    batch_size = 1
    
    print("\nGenerating Data...")
    x_train, y_train = generate_synthetic_data(num_samples=total_samples)
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Hybrid Learning Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    epochs = 20 # 安定すれば20で十分
    
    for epoch in range(epochs):
        epoch_correct = 0
        total_seen = 0
        
        core.train()
        
        for i, (data, target_onehot) in enumerate(loader):
            target_idx = target_onehot.argmax(dim=1)
            metrics = core.autonomous_step(data, target_idx)
            
            is_correct = 1.0 if metrics["reward"] > 0.0 else 0.0
            
            correct_avg = correct_avg * 0.995 + is_correct * 0.005
            epoch_correct += is_correct
            total_seen += 1
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 1000 == 0:
                # Hidden層の状態
                w_hid = core.fast_process.get_ternary_weights()
                conn_hid = float(w_hid.mean().item()) * 100
                
                # Output層の状態
                w_out = core.output_gate.get_ternary_weights()
                conn_out = float(w_out.mean().item()) * 100
                
                out_spikes = metrics["output_spike_count"]
                
                print(f"Epoch {epoch+1:2d} [{i:4d}/{total_samples}] - "
                      f"Acc(MA): {correct_avg*100:.1f}% | "
                      f"Conn(H): {conn_hid:.1f}% | "
                      f"Conn(O): {conn_out:.1f}% | "
                      f"Spikes: {out_spikes:.1f}")
        
        epoch_acc = epoch_correct / total_samples * 100
        print(f"--- Epoch {epoch+1} Final Accuracy: {epoch_acc:.2f}% ---")
        
        if epoch_acc > 95.0:
            print("Target Accuracy Reached. Early Stopping.")
            break

    print("\nRunning Final Evaluation...")
    core.eval()
    test_samples = 1000
    x_test, y_test = generate_synthetic_data(num_samples=test_samples)
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    correct_test = 0
    with torch.no_grad():
        for i in range(test_samples):
            inp = x_test[i:i+1]
            tgt = y_test[i:i+1].argmax(dim=1)
            
            out = core(inp) 
            
            if out.dim() == 1:
                out = out.unsqueeze(0)
            
            pred = out.argmax(dim=1)
            
            if pred.item() == tgt.item() and out.sum().item() > 0:
                correct_test += 1
                
    print(f"Final Test Accuracy: {correct_test/test_samples*100:.2f}%")

if __name__ == "__main__":
    run_simulation()
