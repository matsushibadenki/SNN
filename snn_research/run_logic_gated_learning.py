# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 論理ゲート駆動・自律学習シミュレーション
# 目的: 行列演算とBPを完全に排除した状態で、モデルがデータに適合していく過程を実証する。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore
from snn_research.utils.efficiency_profiler import print_efficiency_report

def generate_synthetic_data(num_samples: int = 1000, in_features: int = 784, out_features: int = 10):
    """シミュレーション用のスパイクデータ（二値）を生成"""
    x = (torch.randn(num_samples, in_features) > 0.5).float()
    # シンプルな決定論的ルールでラベルを生成
    y = (x.sum(dim=1) % out_features).long()
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    # 1. ハイパーパラメータ設定
    IN_FEATURES = 784
    HIDDEN_FEATURES = 256
    OUT_FEATURES = 10
    EPOCHS = 5
    
    print(f"Initializing Hybrid Neuromorphic Core...")
    core = HybridNeuromorphicCore(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES)
    
    # 2. 効率レポートの初期表示
    print_efficiency_report(core)
    
    # 3. データの準備
    x_train, y_train = generate_synthetic_data()
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True) # 局所学習のためバッチサイズ1を推奨
    
    print("\nStarting Autonomous Learning (No Backprop, No GEMM)...")
    
    for epoch in range(EPOCHS):
        total_error = 0.0
        spike_count = 0.0
        
        for i, (data, target) in enumerate(loader):
            # 外部オプティマイザ(Adam等)を使わず、コア自体が学習
            metrics = core.autonomous_step(data, target)
            
            total_error += metrics["prediction_error"]
            spike_count += metrics["output_spike_count"]
            
            if i % 200 == 0 and i > 0:
                print(f"Epoch {epoch+1} [{i}/1000] - Avg Error: {total_error/(i+1):.4f}")

    print("\nSimulation Completed.")
    # 学習後の効率（スパース性など）を再評価する余地あり

if __name__ == "__main__":
    run_simulation()