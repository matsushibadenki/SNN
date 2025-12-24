# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Final: 高精度達成版)

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 5000, in_features: int = 784, out_features: int = 10):
    """
    複雑な論理パターンの生成 (Modulo演算)
    """
    # スパースなバイナリ入力 (密度 50%)
    x = (torch.randn(num_samples, in_features) > 0.0).float()
    y = []
    
    # ターゲット領域: 200-260 (60ビット) の合計 mod 10
    # これは線形分離不可能な非線形タスク
    target_region = x[:, 200:260]
    sum_val = target_region.sum(dim=1).long()
    y = sum_val % out_features
    
    return x, y

def run_simulation():
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    # パラメータ設定
    IN_FEATURES = 784
    HIDDEN_FEATURES = 4096 # 容量を増やしてSparse表現力を高める
    OUT_FEATURES = 10
    BATCH_SIZE = 128      # バッチサイズを大きくして勾配推定を安定化
    TOTAL_SAMPLES = 30000
    EPOCHS = 50

    # モデル構築
    core = HybridNeuromorphicCore(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(device)
    
    print(f"\nModel initialized with {HIDDEN_FEATURES} hidden neurons.")
    print("Generating High-Fidelity Synthetic Data...")
    
    x_train, y_train = generate_synthetic_data(num_samples=TOTAL_SAMPLES, in_features=IN_FEATURES)
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("\nStarting Forward-Forward / DFA Training Phase...")
    
    moving_avg_acc = 0.1
    
    for epoch in range(EPOCHS):
        epoch_correct = 0
        total_seen = 0
        
        core.train() # モード切替（内部的には常に自律更新）
        
        for i, (data, target) in enumerate(loader):
            # 自律学習ステップ (BPなし)
            metrics = core.autonomous_step(data, target)
            
            # 評価用フォワードパス
            with torch.no_grad():
                out = core(data)
                pred = out.argmax(dim=1)
                correct = (pred == target).float().sum().item()
                epoch_correct += correct
                total_seen += data.size(0)
            
            batch_acc = correct / data.size(0)
            moving_avg_acc = moving_avg_acc * 0.95 + batch_acc * 0.05
            
            if i % 100 == 0:
                # 接続性の確認
                w_hid = core.fast_process.get_ternary_weights()
                # ゼロ以外の重みの割合
                conn_hid = (w_hid != 0).float().mean().item() * 100
                
                print(f"Epoch {epoch+1:2d} [{i*BATCH_SIZE:5d}/{TOTAL_SAMPLES}] - "
                      f"Acc: {batch_acc*100:.1f}% (MA: {moving_avg_acc*100:.1f}%) | "
                      f"Conn: {conn_hid:.1f}% | "
                      f"Spikes: {metrics['output_spike_count']:.1f}")
        
        epoch_acc = epoch_correct / total_seen * 100
        print(f"--- Epoch {epoch+1} Final Accuracy: {epoch_acc:.2f}% ---")
        
        # 早期終了条件
        if epoch_acc > 95.0:
            print(">>> Target Accuracy (95%) Reached. Optimization Complete.")
            break

    print("\nRunning Final Evaluation on Test Set...")
    core.eval()
    
    TEST_SAMPLES = 5000
    x_test, y_test = generate_synthetic_data(num_samples=TEST_SAMPLES, in_features=IN_FEATURES)
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    test_correct = 0
    with torch.no_grad():
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        for data, target in test_loader:
            out = core(data)
            pred = out.argmax(dim=1)
            test_correct += (pred == target).float().sum().item()
                
    final_acc = test_correct / TEST_SAMPLES * 100
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    
    if final_acc > 90.0:
        print("SUCCESS: The Logic-Gated SNN has successfully learned the modulo logic task without Backpropagation!")
    else:
        print("WARNING: Accuracy is below target. Consider increasing Hidden Layer size or Epochs.")

if __name__ == "__main__":
    run_simulation()
