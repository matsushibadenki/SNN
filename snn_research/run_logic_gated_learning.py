# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 論理ゲート駆動・自律学習シミュレーション (長期収束版)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore
from snn_research.utils.efficiency_profiler import print_efficiency_report

def generate_synthetic_data(num_samples: int = 1000, in_features: int = 784, out_features: int = 10):
    # パターンの差別化をさらに強調
    x = (torch.randn(num_samples, in_features) > 0.4).float()
    # 決定論的な分類ルールを明確化
    y = (x[:, :in_features//8].sum(dim=1).long() % out_features)
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    IN_FEATURES = 784
    HIDDEN_FEATURES = 256
    OUT_FEATURES = 10
    
    core = HybridNeuromorphicCore(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES)
    print_efficiency_report(core)
    
    x_train, y_train = generate_synthetic_data()
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("\nStarting Long-term Autonomous Learning...")
    
    moving_avg_error = 0.1 # 初期値
    for epoch in range(10): # 長めに回して収束を確認
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            e = metrics["prediction_error"]
            moving_avg_error = moving_avg_error * 0.99 + e * 0.01
            
            if i % 200 == 0:
                conn_rate = float(core.fast_process.get_ternary_weights().mean().item()) * 100
                print(f"Epoch {epoch+1:2d} [{i:4d}/1000] - MA-Error: {moving_avg_error:.8f} | Conn: {conn_rate:.1f}%")

    print("\nSimulation Completed.")

if __name__ == "__main__":
    run_simulation()
