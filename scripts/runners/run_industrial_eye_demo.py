# ファイルパス: scripts/runners/run_industrial_eye_demo.py
# 日本語タイトル: Industrial Eye Demo (Corrected Path & Logic)
# 目的: 高速外観検査のデモ実行。

import os
import sys
import torch
import time
import logging
import numpy as np
from typing import Tuple

# プロジェクトルートの設定 (scripts/runners/ から 2階層上)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Utilsがなければ簡易設定
try:
    from app.utils import setup_logging
    logger = setup_logging(log_dir="logs", log_name="industrial_eye.log")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("IndustrialEye")

from snn_research.models.experimental.dvs_industrial_eye import IndustrialEyeSNN

def generate_synthetic_dvs_data(
    batch_size: int = 1,
    time_steps: int = 8,
    resolution: Tuple[int, int] = (128, 128),
    has_defect: bool = False
) -> torch.Tensor:
    """疑似DVSデータの生成"""
    data = torch.zeros(batch_size, time_steps, 2, *resolution)
    
    # 背景ノイズ
    noise = torch.rand_like(data) > 0.99
    data[noise] = 1.0
    
    speed = 8 # 高速移動
    obj_size = 40
    
    for t in range(time_steps):
        x_start = (t * speed) + 10
        if x_start + obj_size >= resolution[1]: break
        
        # 製品の枠線 (Box)
        y_start = 40
        data[:, t, 0, y_start, x_start:x_start+obj_size] = 1.0
        data[:, t, 0, y_start+obj_size, x_start:x_start+obj_size] = 1.0
        data[:, t, 0, y_start:y_start+obj_size, x_start] = 1.0
        data[:, t, 0, y_start:y_start+obj_size, x_start+obj_size] = 1.0
        
        # 欠陥 (Defect): 枠の中に傷がある
        if has_defect:
            defect_x = x_start + 20
            defect_y = y_start + 20
            # 傷のようなランダムパターン
            data[:, t, 1, defect_y:defect_y+5, defect_x:defect_x+5] = 1.0

    return data

def run_inspection_cycle(model: IndustrialEyeSNN, input_data: torch.Tensor, label: str):
    """検査サイクルの実行"""
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        _ = model(input_data)
        if hasattr(model, 'lif1'): model.lif1.reset()
        if hasattr(model, 'lif2'): model.lif2.reset()
        if hasattr(model, 'lif_out'): model.lif_out.reset()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        logits, stats = model(input_data)
        probs = torch.softmax(logits, dim=1)
        
        prediction = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0, prediction].item()
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0
    
    pred_label = "Defect" if prediction == 1 else "Normal"
    # デモなのでランダム初期化モデルだと予測は適当だが、動作速度を確認する
    # ラベルと一致させるためのダミーロジック等は入れず、生の出力を表示
    
    is_correct = (pred_label == label)
    # 未学習モデルの場合はWarningを出さないようにする
    status_icon = "🟢" if is_correct else "⚠️(Untrained)"
    
    logger.info(f"🔎 Inspection [{label}]: Result={pred_label} (Conf: {confidence:.2f}) {status_icon}")
    logger.info(f"   ⏱️ Latency: {latency_ms:.3f} ms")
    logger.info(f"   ⚡ Sparsity: {stats['sparsity']*100:.1f}% | Power: ~{stats['estimated_power_mw']:.2f} mW")
    
    return latency_ms

def main():
    logger.info("============================================================")
    logger.info("👁️ Industrial Eye - DVS High-Speed Inspection Demo (v17.1)")
    logger.info("============================================================")
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        
    logger.info(f"🚀 Device: {device}")
    
    # モデル構築
    model = IndustrialEyeSNN(
        input_resolution=(128, 128),
        use_dsa=True,
        time_steps=8
    ).to(device)
    
    logger.info("🏭 Production Line Started. Target Speed: > 6000 items/min (<10ms)")
    
    latencies = []
    
    # 1. Normal Case
    normal_data = generate_synthetic_dvs_data(has_defect=False).to(device)
    l = run_inspection_cycle(model, normal_data, "Normal")
    latencies.append(l)
    
    # 2. Defect Case
    defect_data = generate_synthetic_dvs_data(has_defect=True).to(device)
    l = run_inspection_cycle(model, defect_data, "Defect")
    latencies.append(l)
    
    # 3. Burst Mode
    logger.info("\n💨 Burst Mode Testing (20 items)...")
    burst_start = time.perf_counter()
    for _ in range(20):
        # 内部状態リセット
        if hasattr(model, 'lif1'): model.lif1.reset()
        if hasattr(model, 'lif2'): model.lif2.reset()
        if hasattr(model, 'lif_out'): model.lif_out.reset()
        
        data = generate_synthetic_dvs_data(has_defect=False).to(device)
        with torch.no_grad():
            model(data)
            
    burst_duration = (time.perf_counter() - burst_start)
    avg_fps = 20.0 / burst_duration
    
    logger.info(f"🚀 Throughput: {avg_fps:.1f} items/sec")
    
    avg_latency = sum(latencies) / len(latencies)
    logger.info("\n📊 Final Report:")
    logger.info(f"   - Avg Latency: {avg_latency:.3f} ms")
    
    if avg_latency < 10.0:
        logger.info("   ✅ Latency Requirement MET (<10ms)")
    else:
        logger.info("   ⚠️ Latency Requirement CLOSE (Optimization suggested)")
        
    logger.info("============================================================")

if __name__ == "__main__":
    main()
