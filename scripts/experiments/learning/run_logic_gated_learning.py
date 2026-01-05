# ファイルパス: scripts/run_logic_gated_learning.py
# タイトル: Phase-Critical SCAL 検証実験
# 内容: 科学的に正当な実装での性能評価

import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import random
import numpy as np
from typing import Tuple, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f"Seed set to: {seed}")

def generate_synthetic_data(
    num_samples: int,
    in_features: int,
    out_features: int,
    prototypes: torch.Tensor,
    noise_level: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic noisy data
    
    Args:
        num_samples: サンプル数
        in_features: 入力次元
        out_features: クラス数
        prototypes: クラスプロトタイプ [out_features, in_features]
        noise_level: ノイズ率 (0.0-0.5)
    
    Returns:
        x: 入力データ [num_samples, in_features]
        y: ラベル [num_samples]
    """
    device = prototypes.device
    
    # ラベル生成
    labels = torch.randint(0, out_features, (num_samples,), device=device)
    
    # パターン取得
    patterns = prototypes[labels]
    
    # ビット反転ノイズ
    noise_mask = (torch.rand(num_samples, in_features, device=device) < noise_level).float()
    x = torch.abs(patterns - noise_mask)
    
    return x, labels

class ExperimentLogger:
    """実験ログの記録"""
    
    def __init__(self):
        self.history = []
    
    def log_epoch(self, epoch: int, stage: str, metrics: Dict):
        entry = {'epoch': epoch, 'stage': stage, **metrics}
        self.history.append(entry)
    
    def print_header(self):
        print(f"{'Epoch':<6} | {'Stage':<20} | {'Acc':<7} | {'Loss':<7} | "
              f"{'SpkRate':<8} | {'V_th':<7} | {'Var':<7} | {'Temp':<7} | {'Time'}")
        print("-" * 110)
    
    def print_epoch(self, entry: Dict):
        print(f"{entry['epoch']:<6} | {entry['stage']:<20} | "
              f"{entry['accuracy']*100:6.2f}% | {entry['loss']:7.5f} | "
              f"{entry['spike_rate']*100:7.2f}% | {entry['v_th']:7.4f} | "
              f"{entry['variance']:7.4f} | {entry['temperature']:7.4f} | "
              f"{entry['elapsed']:.0f}s")

def run_experiment():
    """Phase-Critical SCAL の実験"""
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}\n")
    
    # === パラメータ設定 ===
    IN_FEATURES = 784
    OUT_FEATURES = 10
    TRAIN_SAMPLES = 50000
    TEST_SAMPLES = 10000
    
    # Phase-Critical パラメータ
    GAMMA = 0.015          # 閾値適応率
    V_TH_INIT = 0.6        # 初期閾値
    TARGET_SPIKE_RATE = 0.15  # 目標発火率 15%
    
    # カリキュラム設定
    curriculum = [
        {'noise': 0.10, 'epochs': 10, 'lr': 0.05, 'batch': 2048},
        {'noise': 0.20, 'epochs': 10, 'lr': 0.03, 'batch': 2048},
        {'noise': 0.30, 'epochs': 15, 'lr': 0.02, 'batch': 4096},
        {'noise': 0.40, 'epochs': 20, 'lr': 0.01, 'batch': 4096},
        {'noise': 0.45, 'epochs': 30, 'lr': 0.005, 'batch': 8192},
        {'noise': 0.47, 'epochs': 20, 'lr': 0.002, 'batch': 8192},
    ]
    
    # === モデル初期化 ===
    from snn_research.core.layers.logic_gated_snn import PhaseCriticalSCAL
    
    model = PhaseCriticalSCAL(
        IN_FEATURES, OUT_FEATURES,
        mode='readout',
        gamma=GAMMA,
        v_th_init=V_TH_INIT,
        target_spike_rate=TARGET_SPIKE_RATE
    ).to(device)
    
    # プロトタイプ生成
    prototypes = (torch.randn(OUT_FEATURES, IN_FEATURES, device=device) > 0.0).float()
    
    print("=== Phase-Critical SCAL ===")
    print("Parameters:")
    print(f"  γ (threshold adaptation rate): {GAMMA}")
    print(f"  V_th_init: {V_TH_INIT}")
    print(f"  Target spike rate: {TARGET_SPIKE_RATE*100:.1f}%")
    print(f"  Input dimension: {IN_FEATURES}")
    print(f"  Output classes: {OUT_FEATURES}\n")
    
    logger = ExperimentLogger()
    logger.print_header()
    
    start_time = time.time()
    epoch_counter = 0
    
    # === カリキュラム学習 ===
    for stage_idx, stage in enumerate(curriculum):
        noise_level = stage['noise']
        stage_epochs = stage['epochs']
        learning_rate = stage['lr']
        batch_size = stage['batch']
        
        stage_name = f"Noise {noise_level:.2f}"
        
        for ep in range(stage_epochs):
            epoch_counter += 1
            
            # データ生成
            x_train, y_train = generate_synthetic_data(
                TRAIN_SAMPLES, IN_FEATURES, OUT_FEATURES,
                prototypes, noise_level
            )
            
            dataset = TensorDataset(x_train, y_train)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # 訓練
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            total_samples = 0
            
            for batch_x, batch_y in loader:
                # Forward
                result = model(batch_x)
                output = result['output']
                
                # Loss計算
                target_onehot = torch.zeros_like(output)
                target_onehot.scatter_(1, batch_y.unsqueeze(1), 1.0)
                loss = F.mse_loss(output, target_onehot)
                
                # Accuracy
                pred = output.argmax(dim=1)
                correct = (pred == batch_y).sum().item()
                
                # Plasticity update
                model.update_plasticity(batch_x, result, batch_y, learning_rate)
                
                epoch_loss += loss.item() * batch_x.size(0)
                epoch_correct += correct
                total_samples += batch_x.size(0)
            
            # メトリクス
            avg_loss = epoch_loss / total_samples
            accuracy = epoch_correct / total_samples
            
            metrics = model.get_phase_critical_metrics()
            elapsed = time.time() - start_time
            
            log_entry = {
                'epoch': epoch_counter,
                'stage': stage_name,
                'accuracy': accuracy,
                'loss': avg_loss,
                'spike_rate': metrics['spike_rate'],
                'v_th': metrics['mean_threshold'],
                'variance': metrics['mean_variance'],
                'temperature': metrics['temperature'],
                'elapsed': elapsed
            }
            
            logger.log_epoch(epoch_counter, stage_name, log_entry)
            
            # 5エポックごとに表示
            if epoch_counter % 5 == 0 or ep == stage_epochs - 1:
                logger.print_epoch(log_entry)
    
    print("\n=== Training Complete ===\n")
    
    # === ロバスト性評価 ===
    print("=== Robustness Evaluation ===")
    print(f"{'Noise':<7} | {'Acc':<8} | {'Loss':<8} | {'SpkRate':<9} | "
          f"{'V_th':<8} | {'Status'}")
    print("-" * 70)
    
    model.eval()
    test_noise_levels = [0.10, 0.20, 0.30, 0.40, 0.45, 0.48, 0.50]
    
    results = []
    
    for noise in test_noise_levels:
        x_test, y_test = generate_synthetic_data(
            TEST_SAMPLES, IN_FEATURES, OUT_FEATURES,
            prototypes, noise
        )
        
        with torch.no_grad():
            # 複数回実行して平均（確率的発火のため）
            n_trials = 5
            accuracies = []
            losses = []
            spike_rates = []
            
            for _ in range(n_trials):
                test_dataset = TensorDataset(x_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=4096)
                
                correct = 0
                total_loss = 0.0
                total_spikes = 0.0
                total = 0
                
                for batch_x, batch_y in test_loader:
                    result = model(batch_x)
                    output = result['output']
                    spikes = result['spikes']
                    
                    target_onehot = torch.zeros_like(output)
                    target_onehot.scatter_(1, batch_y.unsqueeze(1), 1.0)
                    
                    loss = F.mse_loss(output, target_onehot)
                    pred = output.argmax(dim=1)
                    
                    correct += (pred == batch_y).sum().item()
                    total_loss += loss.item() * batch_x.size(0)
                    total_spikes += spikes.mean().item() * batch_x.size(0)
                    total += batch_x.size(0)
                
                accuracies.append(correct / total)
                losses.append(total_loss / total)
                spike_rates.append(total_spikes / total)
        
        # 平均と標準偏差
        acc_mean = np.mean(accuracies)
        acc_std = np.std(accuracies)
        loss_mean = np.mean(losses)
        spike_mean = np.mean(spike_rates)
        
        metrics = model.get_phase_critical_metrics()
        v_th = metrics['mean_threshold']
        
        # Status判定
        if acc_mean > 0.98:
            status = "Excellent"
        elif acc_mean > 0.90:
            status = "Good"
        elif acc_mean > 0.80:
            status = "Acceptable"
        elif noise >= 0.48:
            status = "Near Limit"
        else:
            status = "Weak"
        
        if noise == 0.45 and acc_mean > 0.88:
            status = "SOTA ✓"
        
        print(f"{noise:<7.2f} | {acc_mean*100:7.2f}% | {loss_mean:8.5f} | "
              f"{spike_mean*100:8.2f}% | {v_th:8.4f} | {status}")
        
        results.append({
            'noise': noise,
            'accuracy': acc_mean,
            'accuracy_std': acc_std,
            'loss': loss_mean,
            'spike_rate': spike_mean
        })
    
    print("\n=== Experiment Complete ===")
    
    # サマリー
    print("\nKey Findings:")
    for r in results:
        if r['noise'] in [0.30, 0.45, 0.48]:
            print(f"  Noise {r['noise']:.2f}: "
                  f"{r['accuracy']*100:.2f}% ± {r['accuracy_std']*100:.2f}% "
                  f"(Spike Rate: {r['spike_rate']*100:.1f}%)")

if __name__ == "__main__":
    run_experiment()
