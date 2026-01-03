# ファイルパス: scripts/verify_scalability.py
# Title: Scalability Verification Script
# 機能: モデル規模を拡大した際の影響(レイテンシ、メモリ、パラメータ数)を計測する。

import torch
import time
import os
import sys
import yaml
import psutil

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from snn_research.models.transformer.dsa_transformer import SpikingDSATransformer

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def verify_scalability(config_path):
    print(f"🚀 Loading Config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    
    print(f"   Device: {device}")
    
    # 1. モデル構築 & パラメータ数計測
    print("\n📦 Building Scaled Model...")
    mem_before = get_memory_usage()
    
    # Configからパラメータ抽出
    m_conf = config['model']
    model = SpikingDSATransformer(
        input_dim=m_conf['d_model'], # 入力次元も合わせる
        d_model=m_conf['d_model'],
        num_heads=m_conf['num_heads'],
        num_layers=m_conf['num_layers'],
        dim_feedforward=m_conf['dim_feedforward'],
        time_window=m_conf['time_window'],
        use_bitnet=m_conf.get('use_bitnet', False)
    ).to(device)
    
    mem_after = get_memory_usage()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model Built Successfully.")
    print(f"   📊 Total Parameters: {param_count:,}")
    print(f"   💾 Memory Footprint: {mem_after - mem_before:.2f} MB")
    
    # 2. 推論レイテンシ計測 (T=1 Real-time Mode)
    print("\n⚡ Measuring Inference Latency (T=1)...")
    model.eval()
    
    # ウォームアップ
    dummy_input = torch.randn(1, 1, m_conf['d_model']).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # 計測
    start_time = time.time()
    steps = 100
    for _ in range(steps):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
        
    avg_latency = (time.time() - start_time) * 1000 / steps
    print(f"   ⏱️  Average Latency: {avg_latency:.2f} ms")
    
    # 判定
    target_latency = 10.0 # ms
    if avg_latency < target_latency:
        print(f"\n✅ SUCCESS: Scaled model is FAST enough! (< {target_latency}ms)")
    else:
        print(f"\n⚠️  WARNING: Latency exceeds target. Optimization needed.")

if __name__ == "__main__":
    config_path = "configs/models/phase2_scaled.yaml"
    verify_scalability(config_path)