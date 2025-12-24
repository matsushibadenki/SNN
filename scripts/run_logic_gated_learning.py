# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Final: 高精度達成・CPU最適化版)

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 5000, in_features: int = 784, out_features: int = 10):
    """
    複雑な論理パターンの生成 (Modulo演算タスク)
    """
    # 0.0より大なら1、それ以外0 (密度50%のスパース入力)
    x = (torch.randn(num_samples, in_features) > 0.0).float()
    
    # ターゲット領域: 200-260 (60ビット) の合計 mod 10
    # この非線形性は単純な線形分類器では解けない
    target_region = x[:, 200:260]
    sum_val = target_region.sum(dim=1).long()
    y = sum_val % out_features
    
    return x, y

def run_simulation():
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    # パラメータ設定 (CPU向けに調整)
    IN_FEATURES = 784
    # 4096はCPUには重すぎるため2048に調整。これでも十分な表現力があります。
    HIDDEN_FEATURES = 2048 
    OUT_FEATURES = 10
    BATCH_SIZE = 128
    TOTAL_SAMPLES = 20000
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
    print(f"Target: >90% Accuracy. Max Epochs: {EPOCHS}")
    
    moving_avg_acc = 0.1
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_correct = 0
        total_seen = 0
        
        core.train()
        
        for i, (data, target) in enumerate(loader):
            # 自律学習ステップ (BPなし)
            metrics = core.autonomous_step(data, target)
            
            # 精度確認用フォワードパス
            with torch.no_grad():
                out = core(data)
                pred = out.argmax(dim=1)
                correct = (pred == target).float().sum().item()
                epoch_correct += correct
                total_seen += data.size(0)
            
            batch_acc = correct / data.size(0)
            moving_avg_acc = moving_avg_acc * 0.95 + batch_acc * 0.05
            
            # ログ表示間隔
            if i % 50 == 0:
                # 接続性の確認
                w_hid = core.fast_process.get_ternary_weights()
                conn_hid = (w_hid != 0).float().mean().item() * 100
                
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:2d} [{i*BATCH_SIZE:5d}/{TOTAL_SAMPLES}] "
                      f"Acc: {batch_acc*100:4.1f}% (MA: {moving_avg_acc*100:4.1f}%) | "
                      f"Conn: {conn_hid:4.1f}% | "
                      f"Spikes: {metrics['output_spike_count']:.1f} | "
                      f"Time: {elapsed:.0f}s")
        
        epoch_acc = epoch_correct / total_seen * 100
        print(f"--- Epoch {epoch+1} Final Accuracy: {epoch_acc:.2f}% ---")
        
        # 早期終了条件
        if epoch_acc > 95.0:
            print(">>> Target Accuracy (95%) Reached. Optimization Complete.")
            break
            
        # 安全装置: 5エポック経過しても精度が15%以下の場合は設定ミスとみなして中断
        if epoch >= 5 and epoch_acc < 15.0:
            print(">>> FAILURE DETECTED: Model is not learning (Accuracy < 15%). Stopping.")
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
        print("SUCCESS: The Logic-Gated SNN has successfully learned the modulo logic task!")
    else:
        print("RESULT: Training finished. Please adjust hyperparameters if accuracy is insufficient.")

if __name__ == "__main__":
    run_simulation()
