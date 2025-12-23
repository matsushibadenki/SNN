# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 論理ゲート駆動・自律学習シミュレーション (精度向上版)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 1000, in_features: int = 784, out_features: int = 10):
    # スパース入力を生成 (MNISTのような入力を想定)
    x = (torch.randn(num_samples, in_features) > 1.5).float()
    # 入力の特定セグメントに基づいた論理ターゲット (より複雑な相関)
    # 200:300の範囲の合計値を元に決定
    sum_val = x[:, 200:300].sum(dim=1).long()
    y = sum_val % out_features
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    # 512隠れユニット構成のコア
    core = HybridNeuromorphicCore(784, 512, 10)
    x_train, y_train = generate_synthetic_data(num_samples=2000)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Optimization Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    
    for epoch in range(15):  # 収束のためにエポックを少し増やす
        for i, (data, target) in enumerate(loader):
            # 自律学習ステップの実行
            metrics = core.autonomous_step(data, target)
            
            # 正解判定: 予測誤差が小さく、報酬が得られているか
            is_correct = 1.0 if metrics["reward"] > 0.5 else 0.0
            correct_avg = correct_avg * 0.98 + is_correct * 0.02
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.98 + e * 0.02
            
            if i % 200 == 0:
                # 接続率の取得
                conn = float(core.fast_process.get_ternary_weights().mean().item()) * 100
                # 習熟度の取得
                prof = float(core.fast_process.proficiency.item())
                print(f"Epoch {epoch+1:2d} [{i:4d}/2000] - Error: {ma_error:.4f} | Conn: {conn:.1f}% | Acc(MA): {correct_avg*100:.1f}% | Prof: {prof:.3f}")

if __name__ == "__main__":
    run_simulation()
