# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (Final: パターン認識・対照学習版)

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
    SNNが得意とする「ノイズのあるパターン認識タスク」を生成
    各クラスに対して固有の「プロトタイプ（原型）」を作成し、
    それをノイズで崩したものを入力とする。
    """
    # 1. 各クラスのプロトタイプを作成 (ランダムな疎なパターン)
    # クラスごとに固定の「正解パターン」を決める
    prototypes = (torch.randn(out_features, in_features) > 0.5).float()
    
    x_data = []
    y_data = []
    
    for _ in range(num_samples):
        # ランダムにクラスを選択
        label = torch.randint(0, out_features, (1,)).item()
        
        # プロトタイプを取得
        pattern = prototypes[label].clone()
        
        # 2. ノイズ注入 (ビット反転)
        # 20%の確率でビットを反転させ、不確実性を持たせる
        noise_mask = (torch.rand(in_features) < 0.2).float()
        # XOR的な操作でビット反転 (0->1, 1->0)
        noisy_pattern = torch.abs(pattern - noise_mask)
        
        x_data.append(noisy_pattern)
        y_data.append(label)
    
    x = torch.stack(x_data)
    y = torch.tensor(y_data, dtype=torch.long)
    
    return x, y

def run_simulation():
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    # パラメータ設定
    IN_FEATURES = 784
    HIDDEN_FEATURES = 2048
    OUT_FEATURES = 10
    BATCH_SIZE = 64 # バッチサイズを小さくして更新頻度を上げる
    TOTAL_SAMPLES = 10000 # サンプル数を適正化
    EPOCHS = 20 

    # モデル構築
    core = HybridNeuromorphicCore(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(device)
    
    print(f"\nModel initialized with {HIDDEN_FEATURES} hidden neurons.")
    print("Generating Pattern Recognition Data (Prototypes + Noise)...")
    
    x_train, y_train = generate_synthetic_data(num_samples=TOTAL_SAMPLES, in_features=IN_FEATURES)
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("\nStarting Contrastive Hebbian Learning Phase...")
    print(f"Target: >90% Accuracy. Max Epochs: {EPOCHS}")
    
    moving_avg_acc = 0.1
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_correct = 0
        total_seen = 0
        
        core.train()
        
        for i, (data, target) in enumerate(loader):
            # 自律学習ステップ (Contrastive Learning)
            metrics = core.autonomous_step(data, target)
            
            # 精度確認
            with torch.no_grad():
                out = core(data)
                pred = out.argmax(dim=1)
                correct = (pred == target).float().sum().item()
                epoch_correct += correct
                total_seen += data.size(0)
            
            batch_acc = correct / data.size(0)
            moving_avg_acc = moving_avg_acc * 0.9 + batch_acc * 0.1
            
            # ログ表示
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
            
    print("\nRunning Final Evaluation on Test Set...")
    core.eval()
    
    TEST_SAMPLES = 2000
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

if __name__ == "__main__":
    run_simulation()
