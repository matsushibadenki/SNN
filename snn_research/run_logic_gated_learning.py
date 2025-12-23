# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 論理ゲート駆動・自律学習シミュレーション (知能定着版)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 1000, in_features: int = 784, out_features: int = 10):
    # 情報の密度を最適化
    x = (torch.randn(num_samples, in_features) > 1.0).float()
    y = (x[:, :50].sum(dim=1).long() % out_features)
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    core = HybridNeuromorphicCore(784, 256, 10)
    x_train, y_train = generate_synthetic_data()
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Consolidation Mode)...")
    
    ma_error = 0.1
    correct_avg = 0.1 # Accuracyの移動平均
    
    for epoch in range(10):
        correct_count = 0
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            
            # Accuracyの計算
            is_correct = 1.0 if metrics["reward"] > 1.0 else 0.0
            correct_avg = correct_avg * 0.99 + is_correct * 0.01
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 200 == 0:
                conn = float(core.fast_process.get_ternary_weights().mean().item()) * 100
                print(f"Epoch {epoch+1:2d} [{i:4d}/1000] - Error: {ma_error:.6f} | Conn: {conn:.1f}% | Acc(MA): {correct_avg*100:.1f}%")

    print("\nSimulation Completed.")

if __name__ == "__main__":
    run_simulation()
