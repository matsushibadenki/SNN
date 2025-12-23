# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 論理ゲート駆動・自律学習シミュレーション (ダイナミクス強化版)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore
from snn_research.utils.efficiency_profiler import print_efficiency_report

def generate_synthetic_data(num_samples: int = 1000, in_features: int = 784, out_features: int = 10):
    """入力にわずかなノイズを加え、データの多様性を確保"""
    base_x = (torch.randn(num_samples, in_features) > 0.0).float()
    # 5%の確率でビットを反転
    noise = (torch.rand(num_samples, in_features) < 0.05).float()
    x = torch.abs(base_x - noise)
    
    # ラベル生成ロジックの変更
    y = (x[:, :in_features//2].sum(dim=1).long() % out_features)
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    IN_FEATURES = 784
    HIDDEN_FEATURES = 256
    OUT_FEATURES = 10
    EPOCHS = 5
    
    core = HybridNeuromorphicCore(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES)
    print_efficiency_report(core)
    
    x_train, y_train = generate_synthetic_data()
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("\nStarting Autonomous Learning (Homeostatic Dynamics)...")
    
    for epoch in range(EPOCHS):
        total_error = 0.0
        
        for i, (data, target) in enumerate(loader):
            # autonomous_stepの中で、target情報を積極的に利用するよう修正
            metrics = core.autonomous_step(data, target)
            total_error += metrics["prediction_error"]
            
            if i % 200 == 0 and i > 0:
                # 誤差の推移が見えやすいよう、直近の平均を表示
                print(f"Epoch {epoch+1} [{i}/1000] - Current Error: {metrics['prediction_error']:.6f}")

    print("\nSimulation Completed.")

if __name__ == "__main__":
    run_simulation()
