# ファイルパス: scripts/runners/run_industrial_eye_demo.py
# 日本語タイトル: Industrial Eye Demo (High-Speed Inspection) [Fixed Types]
# 目的・内容:
#   ROADMAP v17.0 "Industrial Eye" のPoCデモ。
#   修正: Tensorのインデックス指定時の型不整合(mypy error)を、明示的なintキャストで解消。

import os
import sys
import torch
import time
import logging
import numpy as np
from typing import Tuple

# プロジェクトルートの設定
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.utils import setup_logging
from snn_research.models.experimental.dvs_industrial_eye import IndustrialEyeSNN

logger = setup_logging(log_dir="logs", log_name="industrial_eye.log")

def generate_synthetic_dvs_data(
    batch_size: int = 1,
    time_steps: int = 8,
    resolution: Tuple[int, int] = (128, 128),
    has_defect: bool = False
) -> torch.Tensor:
    """
    DVSデータを模倣したスパイクテンソルを生成する。
    """
    # (B, T, C, H, W)
    data = torch.zeros(batch_size, time_steps, 2, *resolution)
    
    # 背景ノイズ
    noise = torch.rand_like(data) > 0.98
    data[noise] = 1.0
    
    # オブジェクトの移動
    speed = 5
    obj_w, obj_h = 40, 40
    
    for t in range(time_steps):
        x_start = (t * speed) + 20
        if x_start + obj_w >= resolution[1]: break
        
        # 製品のエッジ
        data[:, t, 0, 40:40+obj_h, x_start] = 1.0 
        data[:, t, 0, 40:40+obj_h, x_start+obj_w] = 1.0 
        data[:, t, 0, 40, x_start:x_start+obj_w] = 1.0 
        data[:, t, 0, 40+obj_h, x_start:x_start+obj_w] = 1.0 
        
        if has_defect:
            defect_x = x_start + 15
            defect_y = 55
            for i in range(10):
                data[:, t, 1, defect_y+i, defect_x+i] = 1.0

    return data

def run_inspection_cycle(model: IndustrialEyeSNN, input_data: torch.Tensor, label: str):
    """
    1回の検査サイクルを実行し、パフォーマンスを計測する。
    """
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        _ = model(input_data)
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        logits, stats = model(input_data)
        probs = torch.softmax(logits, dim=1)
        
        # Fix: mypyがTensorのインデックスとして 'int | float' を懸念するため、明示的に int にキャスト
        prediction = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0, prediction].item()
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0
    
    pred_label = "Defect" if prediction == 1 else "Normal"
    correct = (pred_label == label)
    status_icon = "✅" if correct else "❌"
    
    logger.info(f"🔎 Inspection [{label}]: Predicted={pred_label} ({confidence:.2f}) {status_icon}")
    logger.info(f"   ⏱️ Latency: {latency_ms:.3f} ms")
    logger.info(f"   ⚡ Sparsity: {stats['sparsity']*100:.1f}% | Power: ~{stats['estimated_power_mw']:.2f} mW")
    
    return latency_ms, stats['estimated_power_mw']

def main():
    logger.info("============================================================")
    logger.info("👁️ Industrial Eye - DVS High-Speed Inspection Demo (v17.0)")
    logger.info("============================================================")
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info("🚀 Using CUDA GPU for Inspection")
    
    # 1. モデル初期化
    model = IndustrialEyeSNN(
        input_resolution=(128, 128),
        use_dsa=True,
        time_steps=8
    ).to(device)
    
    logger.info("🏭 Production Line Started. Speed: 500 items/min")
    
    # 2. 検査ループ (シミュレーション)
    latencies = []
    powers = []
    
    # Case A: Normal Product
    normal_data = generate_synthetic_dvs_data(has_defect=False).to(device)
    l, p = run_inspection_cycle(model, normal_data, "Normal")
    latencies.append(l)
    powers.append(p)
    
    time.sleep(0.5)
    
    # Case B: Defective Product
    defect_data = generate_synthetic_dvs_data(has_defect=True).to(device)
    l, p = run_inspection_cycle(model, defect_data, "Defect")
    latencies.append(l)
    powers.append(p)
    
    # Case C: High-Speed Burst
    logger.info("\n🚀 Burst Mode Testing (10 items)...")
    burst_start = time.perf_counter()
    for _ in range(10):
        is_defect = np.random.rand() > 0.8
        data = generate_synthetic_dvs_data(has_defect=is_defect).to(device)
        with torch.no_grad():
            model(data)
    burst_duration = (time.perf_counter() - burst_start) * 1000.0
    avg_fps = 10000.0 / burst_duration # 10 items
    
    logger.info(f"💨 Burst finished. Throughput: {avg_fps:.1f} items/sec")

    # 3. 評価
    avg_latency = sum(latencies) / len(latencies)
    avg_power = sum(powers) / len(powers)
    
    logger.info("\n📊 Performance Report:")
    logger.info(f"   - Avg Latency: {avg_latency:.3f} ms (Target: < 10ms)")
    if avg_latency < 10.0:
        logger.info("   ✅ Latency Requirement MET.")
    else:
        logger.warning("   ⚠️ Latency Requirement NOT MET (Optimization needed).")
        
    logger.info(f"   - Avg Power: {avg_power:.2f} mW")
    
    logger.info("============================================================")
    logger.info("🎉 Industrial Eye Demo Completed.")

if __name__ == "__main__":
    main()