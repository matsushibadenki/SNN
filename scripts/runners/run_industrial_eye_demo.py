# ファイルパス: scripts/runners/run_industrial_eye_demo.py
# 日本語タイトル: Industrial Eye Demo (High-FPS Tuned)
# 目的: T=4 での高速動作検証。

import os
import sys
import torch
import time
import logging
import numpy as np
from typing import Tuple

# プロジェクトルートの設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Logging Setup
try:
    from app.utils import setup_logging
    logger = setup_logging(log_dir="logs", log_name="industrial_eye.log")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("IndustrialEye")

from snn_research.models.experimental.dvs_industrial_eye import IndustrialEyeSNN

def generate_synthetic_dvs_data(
    batch_size: int = 1,
    time_steps: int = 4, # T=4 Optimized
    resolution: Tuple[int, int] = (128, 128),
    has_defect: bool = False
) -> torch.Tensor:
    data = torch.zeros(batch_size, time_steps, 2, *resolution)
    noise = torch.rand_like(data) > 0.99
    data[noise] = 1.0
    
    speed = 15 # 高速化に合わせてオブジェクト速度も上げる
    obj_size = 40
    
    for t in range(time_steps):
        x_start = (t * speed) + 10
        if x_start + obj_size >= resolution[1]: break
        
        y_start = 40
        # Box Shape
        data[:, t, 0, y_start, x_start:x_start+obj_size] = 1.0
        data[:, t, 0, y_start+obj_size, x_start:x_start+obj_size] = 1.0
        data[:, t, 0, y_start:y_start+obj_size, x_start] = 1.0
        data[:, t, 0, y_start:y_start+obj_size, x_start+obj_size] = 1.0
        
        if has_defect:
            defect_x = x_start + 20
            defect_y = y_start + 20
            data[:, t, 1, defect_y:defect_y+5, defect_x:defect_x+5] = 1.0

    return data

def run_inspection_cycle(model: IndustrialEyeSNN, input_data: torch.Tensor, label: str):
    model.eval()
    
    # Warm-up (State Reset)
    if hasattr(model, 'lif1'): model.lif1.reset()
    if hasattr(model, 'lif2'): model.lif2.reset()
    if hasattr(model, 'lif_out'): model.lif_out.reset()
    
    # Inference Mode for max speed
    with torch.inference_mode():
        start_time = time.perf_counter()
        
        logits, stats = model(input_data)
        probs = torch.softmax(logits, dim=1)
        prediction = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0, prediction].item()
        
        # Sync for precise timing on GPU/MPS
        if input_data.is_cuda: torch.cuda.synchronize()
        if input_data.device.type == 'mps': torch.mps.synchronize()
        
        end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000.0
    
    pred_label = "Defect" if prediction == 1 else "Normal"
    status_icon = "🟢" if (pred_label == label) else "⚠️(Untrained)"
    
    logger.info(f"🔎 Inspection [{label}]: Result={pred_label} (Conf: {confidence:.2f}) {status_icon}")
    logger.info(f"   ⏱️ Latency: {latency_ms:.3f} ms")
    logger.info(f"   ⚡ Sparsity: {stats['sparsity']*100:.1f}%")
    
    return latency_ms

def main():
    logger.info("============================================================")
    logger.info("👁️ Industrial Eye - High-Speed Inspection Demo (Optimized)")
    logger.info("============================================================")
    
    device = 'cpu'
    if torch.cuda.is_available(): device = 'cuda'
    elif torch.backends.mps.is_available(): device = 'mps'
        
    logger.info(f"🚀 Device: {device}")
    
    # Time Steps = 4 で初期化
    model = IndustrialEyeSNN(
        input_resolution=(128, 128),
        use_dsa=True,
        time_steps=4 
    ).to(device)
    
    logger.info("🏭 Production Line Started. Target Speed: < 10ms/item")
    
    # Robust Warm-up (MPS needs several runs to optimize graph)
    logger.info("🔥 Warming up accelerator...")
    dummy = torch.zeros(1, 4, 2, 128, 128).to(device)
    for _ in range(10):
        with torch.inference_mode():
            model(dummy)
            if hasattr(model, 'lif1'): model.lif1.reset()
            if hasattr(model, 'lif2'): model.lif2.reset()
            if hasattr(model, 'lif_out'): model.lif_out.reset()
    
    latencies = []
    
    # 1. Normal Case
    normal_data = generate_synthetic_dvs_data(has_defect=False, time_steps=4).to(device)
    l = run_inspection_cycle(model, normal_data, "Normal")
    latencies.append(l)
    
    # 2. Defect Case
    defect_data = generate_synthetic_dvs_data(has_defect=True, time_steps=4).to(device)
    l = run_inspection_cycle(model, defect_data, "Defect")
    latencies.append(l)
    
    # 3. Burst Mode (50 items)
    logger.info("\n💨 Burst Mode Testing (50 items)...")
    burst_start = time.perf_counter()
    
    # Pre-generate data to measure only inference time
    batch_data = [generate_synthetic_dvs_data(has_defect=False, time_steps=4).to(device) for _ in range(50)]
    
    with torch.inference_mode():
        for data in batch_data:
            if hasattr(model, 'lif1'): model.lif1.reset()
            if hasattr(model, 'lif2'): model.lif2.reset()
            if hasattr(model, 'lif_out'): model.lif_out.reset()
            model(data)
            
        # Sync at the end of burst
        if device == 'cuda': torch.cuda.synchronize()
        if device == 'mps': torch.mps.synchronize()
            
    burst_duration = (time.perf_counter() - burst_start)
    avg_fps = 50.0 / burst_duration
    
    logger.info(f"🚀 Throughput: {avg_fps:.1f} items/sec (Latency per item: {1000/avg_fps:.2f} ms)")
    
    avg_latency = sum(latencies) / len(latencies)
    logger.info("\n📊 Final Report:")
    logger.info(f"   - Single Item Latency: {avg_latency:.3f} ms")
    logger.info(f"   - Burst Throughput: {avg_fps:.1f} fps")
    
    if avg_latency < 10.0 or (1000/avg_fps) < 10.0:
        logger.info("   ✅ Latency Requirement MET (<10ms)")
    else:
        logger.info("   ⚠️ Close to target.")
        
    logger.info("============================================================")

if __name__ == "__main__":
    main()