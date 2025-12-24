# ファイルパス: scripts/run_logic_gated_learning.py
# 日本語タイトル: 統合最適化・自律学習シミュレーション (高精度・教師ありヘブ学習版)
# 内容: 教師あり信号を用いた強力な学習ループを実行し、90%以上の精度を目指す。

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# プロジェクトルートへのパスを通す (scriptsから実行されることを想定)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.core.hybrid_core import HybridNeuromorphicCore

def generate_synthetic_data(num_samples: int = 5000, in_features: int = 784, out_features: int = 10):
    """
    論理ゲート学習用データ生成
    入力: 12%密度のスパースノイズ
    出力: 入力のインデックス200-250の範囲にある「1」の個数をカウントし、mod 10をとったもの
    """
    # 決定論的な動作のためにシードを固定しない（汎化性能を見るため）
    # ただし学習の安定性のために少しノイズをマイルドにする
    x = (torch.randn(num_samples, in_features) > 1.0).float() # 閾値を1.2 -> 1.0にして少し入力をリッチに
    
    y = []
    for i in range(num_samples):
        # 空間的論理: 特定領域のビットパターン
        # 範囲を少し広げて、タスクのロバスト性を確認
        sum_val = x[i, 200:260].sum().long() 
        val = sum_val % out_features
        y.append(val)
    
    y = torch.stack(y)
    y_onehot = nn.functional.one_hot(y, out_features).float()
    return x, y_onehot

def run_simulation():
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    # Hidden層を増強して、複雑なパターン分離能力(リザーバ計算能力)を高める
    core = HybridNeuromorphicCore(784, 1024, 10).to(device)
    
    total_samples = 5000
    batch_size = 1 # SNNはオンライン学習が基本
    
    print("\nGenerating Data...")
    x_train, y_train = generate_synthetic_data(num_samples=total_samples)
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("\nStarting Autonomous Intelligence Integration (Supervised Hebbian Mode)...")
    
    ma_error = 0.5
    correct_avg = 0.1
    
    # 学習率スケジューリングのような効果を狙い、エポックが進むにつれて安定させる
    epochs = 10 
    
    for epoch in range(epochs):
        epoch_correct = 0
        total_seen = 0
        
        core.train()
        
        for i, (data, target_onehot) in enumerate(loader):
            # ターゲットのインデックスを取得
            target_idx = target_onehot.argmax(dim=1)
            
            metrics = core.autonomous_step(data, target_idx)
            
            # 正解判定 (もっとも発火したニューロン、もしくは発火があった中でターゲットと一致など)
            # ここではシンプルに「ターゲットクラスのニューロンが発火し、かつ他の発火が少ない」などの厳密な評価はSNNでは難しいので
            # 内部の自律的な報酬(reward)が正であれば「正解」とみなす、もしくは予測を確認する
            
            # 簡易評価: 報酬がプラスなら正解とみなす (autonomous_stepの実装に依存)
            is_correct = 1.0 if metrics["reward"] > 0.0 else 0.0
            
            # 移動平均精度の更新
            correct_avg = correct_avg * 0.99 + is_correct * 0.01
            epoch_correct += is_correct
            total_seen += 1
                
            e = metrics["prediction_error"]
            ma_error = ma_error * 0.99 + e * 0.01
            
            if i % 1000 == 0:
                w = core.fast_process.get_ternary_weights()
                conn = float(w.mean().item()) * 100
                v_avg = float(core.fast_process.membrane_potential.abs().mean().item())
                v_th = float(core.fast_process.adaptive_threshold.mean().item())
                out_spikes = metrics["output_spike_count"]
                
                print(f"Epoch {epoch+1:2d} [{i:4d}/{total_samples}] - "
                      f"Acc(MA): {correct_avg*100:.1f}% | "
                      f"Conn: {conn:.1f}% | "
                      f"V_th: {v_th:.1f} | "
                      f"OutSpikes: {out_spikes:.1f}")
        
        epoch_acc = epoch_correct / total_samples * 100
        print(f"--- Epoch {epoch+1} Final Accuracy: {epoch_acc:.2f}% ---")
        
        if epoch_acc > 95.0:
            print("Target Accuracy Reached. Early Stopping.")
            break

    # 最終テスト
    print("\nRunning Final Evaluation...")
    core.eval() # 学習を止めて評価
    test_samples = 1000
    x_test, y_test = generate_synthetic_data(num_samples=test_samples)
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    correct_test = 0
    with torch.no_grad():
        for i in range(test_samples):
            inp = x_test[i:i+1]
            tgt = y_test[i:i+1].argmax(dim=1)
            
            # forwardだけ呼ぶ
            out = core(inp) 
            
            # 一番発火したニューロンを予測とする
            # SNNの出力はスパイク(0/1)の束なので、ここでは単純に発火したかどうかを見るが、
            # 厳密にはレートコーディング等が望ましい。
            # 今回は autonomous_step 側で強い指導が入っているので、正解ラベルのみが発火する理想状態を目指している
            pred = out.argmax(dim=1)
            
            if pred.item() == tgt.item() and out.sum().item() > 0:
                correct_test += 1
                
    print(f"Final Test Accuracy: {correct_test/test_samples*100:.2f}%")

if __name__ == "__main__":
    run_simulation()
