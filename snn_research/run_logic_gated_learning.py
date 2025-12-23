# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 論理ゲート駆動・自律学習シミュレーション (覚醒確認版)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore
from snn_research.utils.efficiency_profiler import print_efficiency_report

def generate_synthetic_data(num_samples: int = 1000, in_features: int = 784, out_features: int = 10):
    # パターンの差別化を明確にする
    x = (torch.randn(num_samples, in_features) > 0.5).float()
    # ターゲットラベルをデータの分散に強く依存させる
    y = (x[:, :in_features//10].sum(dim=1).long() % out_features)
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
    
    print("\nStarting Autonomous Learning: Forced Activation Mode...")
    
    prev_error = 0.0
    for epoch in range(5):
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            current_error = metrics["prediction_error"]
            
            if i % 200 == 0:
                weights = core.fast_process.get_ternary_weights()
                conn_rate = float(weights.mean().item()) * 100
                status = "STAGNANT" if current_error == prev_error else "EVOLVING"
                print(f"Epoch {epoch+1} [{i}/1000] - Error: {current_error:.8f} | Conn: {conn_rate:.1f}% | {status}")
                prev_error = current_error

    print("\nSimulation Completed.")

if __name__ == "__main__":
    run_simulation()
