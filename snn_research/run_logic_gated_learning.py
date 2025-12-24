# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (精度向上・安定化版)
# 内容: 学習プロセスの安定化とログ出力の改善。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 3000, in_features: int = 784, out_features: int = 10):
    # スパース入力を生成 (入力密度を 15% 程度に制御)
    x = (torch.randn(num_samples, in_features) > 1.5).float()
    
    y = []
    for i in range(num_samples):
        # 特定領域のスパイクの有無を判定基準にする（よりSNNが捉えやすいタスク）
        # 200:250 の範囲の合計値をクラスとするが、ノイズ耐性のため少し丸める
        sum_val = x[i, 200:250].sum().long()
        val = sum_val % out_features
        y.append(val)
    
    y = torch.stack(y)
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    # 入力 784, 隠れ層 512, 出力 10
    core = HybridNeuromorphicCore(784, 512, 10)
    total_samples = 3000
    x_train, y_train = generate_synthetic_data(num_samples=total_samples)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Precision Enhancement Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    
    # Epoch数を増やし、じっくり学習させる
    for epoch in range(15):
        epoch_correct = 0
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            
            # 正解判定: 予測誤差が小さく、報酬が得られている
            is_correct = 1.0 if metrics["reward"] > 0.5 else 0.0
            correct_avg = correct_avg * 0.995 + is_correct * 0.005
            epoch_correct += is_correct
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 500 == 0:
                # 隠れ層の状態を取得（LogicGatedSNN層を想定）
                # 注意: HybridNeuromorphicCoreの内部構造に依存
                w = core.fast_process.get_ternary_weights()
                conn = float(w.mean().item()) * 100
                prof = float(core.fast_process.proficiency.item())
                print(f"Epoch {epoch+1:2d} [{i:4d}/{total_samples}] - Error: {ma_error:.4f} | Conn: {conn:.1f}% | Acc(MA): {correct_avg*100:.1f}% | Prof: {prof:.3f}")
        
        print(f"--- End of Epoch {epoch+1} | Epoch Accuracy: {epoch_correct/total_samples*100:.1f}% ---")

if __name__ == "__main__":
    run_simulation()
