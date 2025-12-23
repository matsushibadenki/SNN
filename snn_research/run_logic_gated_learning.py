# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 論理ゲート駆動・自律学習シミュレーション (ダイナミクス安定版)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 1000, in_features: int = 784, out_features: int = 10):
    # スパースな入力 (約10%)
    x = (torch.randn(num_samples, in_features) > 1.2).float()
    
    # 認識対象を少し広げ、信号のロバスト性を高める
    y = (x[:, :40].sum(dim=1).long() % out_features)
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    core = HybridNeuromorphicCore(784, 256, 10)
    x_train, y_train = generate_synthetic_data()
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Stabilized)...")
    
    ma_error = 0.1
    for epoch in range(10):
        correct_count = 0
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            
            if metrics["reward"] > 0:
                correct_count += 1
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 200 == 0:
                conn = float(core.fast_process.get_ternary_weights().mean().item()) * 100
                acc = (correct_count / (i + 1)) * 100
                print(f"Epoch {epoch+1:2d} [{i:4d}/1000] - Error: {ma_error:.6f} | Conn: {conn:.1f}% | Acc: {acc:.1f}%")

if __name__ == "__main__":
    run_simulation()
