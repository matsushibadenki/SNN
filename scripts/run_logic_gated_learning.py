# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Fix: データ一貫性確保版)

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 5000, in_features: int = 784, out_features: int = 10, prototypes=None):
    """
    パターン認識タスク (高密度プロトタイプ + ノイズ)
    
    Args:
        prototypes: 学習時とテスト時で同じパターンを使うために、外部から受け取れるようにする
    Returns:
        x, y, prototypes: 生成データと、使用したプロトタイプを返す
    """
    # プロトタイプが指定されていない場合（初回）のみ生成
    if prototypes is None:
        # 密度50%のランダムパターンを「正解」として定義
        prototypes = (torch.randn(out_features, in_features) > 0.0).float()
    
    x_data = []
    y_data = []
    
    for _ in range(num_samples):
        label = torch.randint(0, out_features, (1,)).item()
        pattern = prototypes[label].clone()
        
        # ノイズ注入 (10%反転)
        # 学習データとテストデータで異なるノイズが乗るが、原型は同じ
        noise = (torch.rand(in_features) < 0.1).float()
        noisy_pattern = torch.abs(pattern - noise)
        
        x_data.append(noisy_pattern)
        y_data.append(label)
    
    x = torch.stack(x_data)
    y = torch.tensor(y_data, dtype=torch.long)
    
    return x, y, prototypes

def run_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    # パラメータ設定
    IN_FEATURES = 784
    HIDDEN_FEATURES = 4096 
    OUT_FEATURES = 10
    BATCH_SIZE = 128
    TOTAL_SAMPLES = 20000
    EPOCHS = 10 # 収束が早いため短縮

    # モデル構築
    core = HybridNeuromorphicCore(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(device)
    
    print(f"\nModel initialized with {HIDDEN_FEATURES} hidden neurons (Liquid State Machine).")
    
    # --- 重要: 学習データの生成 ---
    print("Generating Training Data...")
    # ここで prototypes を受け取る
    x_train, y_train, shared_prototypes = generate_synthetic_data(
        num_samples=TOTAL_SAMPLES, 
        in_features=IN_FEATURES, 
        out_features=OUT_FEATURES,
        prototypes=None # 初回なのでNone
    )
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("\nStarting Readout Training Phase...")
    print(f"Target: >95% Accuracy. Max Epochs: {EPOCHS}")
    
    moving_avg_acc = 0.1
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_correct = 0
        total_seen = 0
        
        core.train()
        
        for i, (data, target) in enumerate(loader):
            # 自律学習 (Output層の重み更新)
            metrics = core.autonomous_step(data, target)
            
            with torch.no_grad():
                out = core(data)
                pred = out.argmax(dim=1)
                correct = (pred == target).float().sum().item()
                epoch_correct += correct
                total_seen += data.size(0)
            
            batch_acc = correct / data.size(0)
            moving_avg_acc = moving_avg_acc * 0.9 + batch_acc * 0.1
            
            if i % 50 == 0:
                # 重みの平均値
                w_out = core.output_gate.get_effective_weights()
                w_mean = w_out.abs().mean().item()
                
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:2d} [{i*BATCH_SIZE:5d}/{TOTAL_SAMPLES}] "
                      f"Acc: {batch_acc*100:4.1f}% (MA: {moving_avg_acc*100:4.1f}%) | "
                      f"W_Mean: {w_mean:.4f} | "
                      f"Time: {elapsed:.0f}s")
        
        epoch_acc = epoch_correct / total_seen * 100
        print(f"--- Epoch {epoch+1} Final Accuracy: {epoch_acc:.2f}% ---")
        
        if epoch_acc > 99.0:
            print(">>> Target Accuracy Reached. Optimization Complete.")
            break
    
    print("\nRunning Final Evaluation...")
    core.eval()
    
    TEST_SAMPLES = 5000
    # --- 重要: テストデータの生成 ---
    # 学習時と同じ shared_prototypes を渡す！これで同じルールでのテストになる
    x_test, y_test, _ = generate_synthetic_data(
        num_samples=TEST_SAMPLES, 
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        prototypes=shared_prototypes 
    )
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
        print("SUCCESS: 90% Accuracy Barrier Broken!")

if __name__ == "__main__":
    run_simulation()
