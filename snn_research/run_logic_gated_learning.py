# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 精密論理認識・自律学習シミュレーション (最終調整版)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 1000, in_features: int = 784, out_features: int = 10):
    # スパース性を調整 (1.0 -> 0.8)
    x = (torch.randn(num_samples, in_features) > 0.8).float()
    
    # 認識対象の空間パターンを少し明確にする
    # 入力の特定セグメント [200:300] を10個の領域に分け、その反応でクラスを決定
    segment = x[:, 200:300].view(num_samples, 10, 10)
    y = segment.sum(dim=2).argmax(dim=1)
    
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    core = HybridNeuromorphicCore(784, 512, 10)
    # 学習サンプル数を増やし、より多くの統計機会を与える
    total_samples = 3000
    x_train, y_train = generate_synthetic_data(num_samples=total_samples)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Precision Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    
    for epoch in range(10):
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            
            # 報酬スレッショルドを調整
            is_correct = 1.0 if metrics["reward"] > 0.5 else 0.0
            # 移動平均の平滑化係数を微調整
            correct_avg = correct_avg * 0.99 + is_correct * 0.01
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 300 == 0:
                conn = float(core.fast_process.get_ternary_weights().mean().item()) * 100
                prof = float(core.fast_process.proficiency.item())
                print(f"Epoch {epoch+1:2d} [{i:4d}/{total_samples}] - Error: {ma_error:.4f} | Conn: {conn:.1f}% | Acc(MA): {correct_avg*100:.1f}% | Prof: {prof:.3f}")

if __name__ == "__main__":
    run_simulation()
