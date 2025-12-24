# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: ポテンシャル学習駆動・自律シミュレーション
# 目的: 壊滅的忘却を克服し、認識精度 90% 以上を安定出力する。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 3000, in_features: int = 784, out_features: int = 10):
    # スパイク入力を生成
    x = (torch.randn(num_samples, in_features) > 1.2).float()
    
    # 認識対象: 受容野を少し重ねることで、SNNに「境界の判断」を学習させる
    y = []
    for i in range(num_samples):
        scores = []
        for c in range(out_features):
            # 各クラスに25ビットの「核」となる領域を割り当て（一部重複）
            start = 200 + c * 15
            end = start + 25
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
    
    print("\nStarting Autonomous Intelligence Integration (Potential Learning Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    
    for epoch in range(12):
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            
            # 成功報酬が得られているかを判定
            is_correct = 1.0 if metrics["reward"] > 0.5 else 0.0
            # 精度表示をより安定させるために平滑化係数を調整
            correct_avg = correct_avg * 0.995 + is_correct * 0.005
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 500 == 0:
                conn = float(core.fast_process.get_ternary_weights().mean().item()) * 100
                prof = float(core.fast_process.proficiency.item())
                # Acc(MA)が着実に上昇し、Connが25%付近で安定することを目指す
                print(f"Epoch {epoch+1:2d} [{i:4d}/{total_samples}] - Error: {ma_error:.4f} | Conn: {conn:.1f}% | Acc(MA): {correct_avg*100:.1f}% | Prof: {prof:.3f}")

if __name__ == "__main__":
    run_simulation()
