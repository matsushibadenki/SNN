# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 正規化ロジック駆動・自律学習シミュレーション
# 目的: シナプス飽和を回避し、90%以上の精度を安定して達成する。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 3000, in_features: int = 784, out_features: int = 10):
    # スパース入力を生成
    x = (torch.randn(num_samples, in_features) > 1.2).float()
    
    # 認識ロジック: 重複のないクリアな空間特徴
    # 10クラスに対し、200:400の範囲を20ビットずつの独立した「受容野」に割り当て
    y = []
    for i in range(num_samples):
        class_scores = []
        for c in range(out_features):
            start = 200 + c * 20
            end = start + 20
            class_scores.append(x[i, start:end].sum())
        y.append(torch.tensor(class_scores).argmax())
    
    y = torch.stack(y)
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    # コアの初期化
    core = HybridNeuromorphicCore(784, 512, 10)
    total_samples = 3000
    x_train, y_train = generate_synthetic_data(num_samples=total_samples)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Normalization Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    
    for epoch in range(10):
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            
            # 報酬から正解を判定 (コア側の報酬設計 0.0 ~ 10.0 前後を想定)
            is_correct = 1.0 if metrics["reward"] > 1.0 else 0.0
            # 指数移動平均で精度を算出
            correct_avg = correct_avg * 0.99 + is_correct * 0.01
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 500 == 0:
                conn = float(core.fast_process.get_ternary_weights().mean().item()) * 100
                prof = float(core.fast_process.proficiency.item())
                print(f"Epoch {epoch+1:2d} [{i:4d}/{total_samples}] - Error: {ma_error:.4f} | Conn: {conn:.1f}% | Acc(MA): {correct_avg*100:.1f}% | Prof: {prof:.3f}")

if __name__ == "__main__":
    run_simulation()
