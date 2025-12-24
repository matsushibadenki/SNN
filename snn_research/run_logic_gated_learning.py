# ファイルパス: snn_research/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (構造固定版)
# 内容: 接続率を安定させ、論理の洗練に集中させる。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 3000, in_features: int = 784, out_features: int = 10):
    # スパース入力 (12% 密度)
    x = (torch.randn(num_samples, in_features) > 1.2).float()
    
    y = []
    for i in range(num_samples):
        # 空間的な特徴（特定範囲の和）
        val = (x[i, 200:250].sum().long()) % out_features
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
    
    print("\nStarting Autonomous Intelligence Integration (Fixed Structure Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    
    for epoch in range(15):
        epoch_correct = 0
        for i, (data, target) in enumerate(loader):
            metrics = core.autonomous_step(data, target)
            
            is_correct = 1.0 if metrics["reward"] > 0.5 else 0.0
            correct_avg = correct_avg * 0.99 + is_correct * 0.01
            epoch_correct += is_correct
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 500 == 0:
                w = core.fast_process.get_ternary_weights()
                conn = float(w.mean().item()) * 100
                prof = float(core.fast_process.proficiency.item())
                v_avg = float(core.fast_process.membrane_potential.abs().mean().item())
                # 閾値の平均を表示
                v_th = float(core.fast_process.adaptive_threshold.mean().item())
                print(f"Epoch {epoch+1:2d} [{i:4d}/{total_samples}] - Error: {ma_error:.4f} | Conn: {conn:.1f}% | Acc(MA): {correct_avg*100:.1f}% | V_avg: {v_avg:.2f} | V_th: {v_th:.1f}")
        
        print(f"--- Epoch {epoch+1} Final Acc: {epoch_correct/total_samples*100:.2f}% | Conn: {conn:.1f}% ---")

if __name__ == "__main__":
    run_simulation()
