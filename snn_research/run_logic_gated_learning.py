# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 空間論理認識・自律学習シミュレーション (局所性強化版)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 2000, in_features: int = 784, out_features: int = 10):
    # コントラストを上げるため閾値を厳しく設定
    x = (torch.randn(num_samples, in_features) > 1.2).float()
    
    # 認識ロジック: 200:400 の範囲のスパイク分布に基づいてクラス分け
    # 空間的に分離された特徴を持たせる
    chunk_size = (400 - 200) // out_features
    y = []
    for i in range(num_samples):
        scores = []
        for c in range(out_features):
            start = 200 + c * chunk_size
            end = start + chunk_size
            scores.append(x[i, start:end].sum())
        y.append(torch.tensor(scores).argmax())
    
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
    
    print("\nStarting Autonomous Intelligence Integration (Spatial Locality Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    
    for epoch in range(10):
        for i, (data, target) in enumerate(loader):
            # 自律学習。内部でupdate_plasticityが呼ばれる。
            # 内部実装がpost_spikesを渡していない場合を考慮し、
            # LogicGatedSNN側でfired_maskの処理を堅牢にしました。
            metrics = core.autonomous_step(data, target)
            
            is_correct = 1.0 if metrics["reward"] > 0 else 0.0
            correct_avg = correct_avg * 0.99 + is_correct * 0.01
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 300 == 0:
                conn = float(core.fast_process.get_ternary_weights().mean().item()) * 100
                prof = float(core.fast_process.proficiency.item())
                # 精度向上の兆しが見えるはず
                print(f"Epoch {epoch+1:2d} [{i:4d}/{total_samples}] - Error: {ma_error:.4f} | Conn: {conn:.1f}% | Acc(MA): {correct_avg*100:.1f}% | Prof: {prof:.3f}")

if __name__ == "__main__":
    run_simulation()
