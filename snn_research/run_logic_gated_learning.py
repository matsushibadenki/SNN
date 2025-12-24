# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 弾性的論理学習・自律シミュレーション (再点火版)
# 目的: 死滅した接続を再活性化し、25%の最適密度で精度 90% を達成する。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 3000, in_features: int = 784, out_features: int = 10):
    # スパイク密度を上げ、ニューロンに最初の刺激を与える
    x = (torch.randn(num_samples, in_features) > 1.0).float()
    
    # 認識対象: 10クラスの空間論理
    y = []
    for i in range(num_samples):
        island_sums = []
        for c in range(out_features):
            start = 200 + c * 20
            end = start + 20
            # ノイズに強い平均値をクラスとする
            island_sums.append(x[i, start:end].sum())
        y.append(torch.tensor(island_sums).argmax())
    
    y = torch.stack(y)
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    core = HybridNeuromorphicCore(784, 512, 10)
    total_samples = 3000
    x_train, y_train = generate_synthetic_data(num_samples=total_samples)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Elastic Re-ignition Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    
    for epoch in range(12):
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            
            # 正解判定: 報酬が出始めているかを確認
            is_correct = 1.0 if metrics["reward"] > 0.5 else 0.0
            correct_avg = correct_avg * 0.99 + is_correct * 0.01
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 500 == 0:
                conn = float(core.fast_process.get_ternary_weights().mean().6f().item() if hasattr(core.fast_process.get_ternary_weights().mean(), 'item') else core.fast_process.get_ternary_weights().mean()) * 100
                # 正しい平均取得のための修正
                conn = float(core.fast_process.get_ternary_weights().mean().item()) * 100
                prof = float(core.fast_process.proficiency.item())
                print(f"Epoch {epoch+1:2d} [{i:4d}/{total_samples}] - Error: {ma_error:.4f} | Conn: {conn:.1f}% | Acc(MA): {correct_avg*100:.1f}% | Prof: {prof:.3f}")

if __name__ == "__main__":
    run_simulation()
