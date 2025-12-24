# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 高精度論理認識・自律学習シミュレーション (最終進化版)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 2000, in_features: int = 784, out_features: int = 10):
    # 明確なバイナリ入力
    x = (torch.randn(num_samples, in_features) > 1.0).float()
    
    # 認識ロジック: 入力の特定領域に「ハミング重み」を持たせ、クラスを決定
    # [200:400]の範囲を10分割し、最もスパイク密度が高いインデックスを正解とする
    # これはSNNが最も得意とする「特徴抽出」タスク
    roi = x[:, 200:400].view(num_samples, out_features, -1)
    y = roi.sum(dim=2).argmax(dim=1)
    
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    core = HybridNeuromorphicCore(784, 512, 10)
    total_samples = 3000
    x_train, y_train = generate_synthetic_data(num_samples=total_samples)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Final Evolutionary Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    
    for epoch in range(15): # 収束を確実にするためエポック数を確保
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            
            # 報酬判定の感度を向上
            is_correct = 1.0 if metrics["reward"] > 0.2 else 0.0
            correct_avg = correct_avg * 0.98 + is_correct * 0.02
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.98 + e * 0.02
            
            if i % 400 == 0:
                conn = float(core.fast_process.get_ternary_weights().mean().item()) * 100
                prof = float(core.fast_process.proficiency.item())
                print(f"Epoch {epoch+1:2d} [{i:4d}/{total_samples}] - Error: {ma_error:.4f} | Conn: {conn:.1f}% | Acc(MA): {correct_avg*100:.1f}% | Prof: {prof:.3f}")

if __name__ == "__main__":
    run_simulation()
