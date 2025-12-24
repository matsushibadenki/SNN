# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (ベスト構成版)
# 内容: 過去の検証で最も結果が良かったロジックを統合。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 3000, in_features: int = 784, out_features: int = 10):
    # スパース入力を生成 (当時のベスト設定)
    x = (torch.randn(num_samples, in_features) > 1.0).float()
    
    # シンプルだが確実な空間論理
    y = []
    for i in range(num_samples):
        # 200:250の範囲の合計をクラスとする
        val = x[i, 200:250].sum().long() % out_features
        y.append(val)
    
    y = torch.stack(y)
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    core = HybridNeuromorphicCore(784, 512, 10)
    total_samples = 3000
    x_train, y_train = generate_synthetic_data(num_samples=total_samples)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Restoring Best Results Loop)...")
    
    ma_error = 0.1
    correct_avg = 0.1
    
    for epoch in range(10):
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            
            # 正解判定: 成功報酬が得られているか
            is_correct = 1.0 if metrics["reward"] > 0 else 0.0
            correct_avg = correct_avg * 0.99 + is_correct * 0.01
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 500 == 0:
                # 接続率の計算 (堅牢な形式)
                w = core.fast_process.get_ternary_weights()
                conn = float(w.mean().item()) * 100
                prof = float(core.fast_process.proficiency.item())
                print(f"Epoch {epoch+1:2d} [{i:4d}/{total_samples}] - Error: {ma_error:.6f} | Conn: {conn:.1f}% | Acc(MA): {correct_avg*100:.1f}% | Prof: {prof:.2f}")

if __name__ == "__main__":
    run_simulation()
