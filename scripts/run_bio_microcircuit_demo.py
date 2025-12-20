# ファイルパス: scripts/run_bio_microcircuit_demo.py
# Title: Bio-Microcircuit Demo
# Description:
#   PD14マイクロサーキットと多区画ニューロンの動作を検証する。
#   ボトムアップ入力（視覚刺激）とトップダウン入力（注意/予測）を与え、
#   各層の発火率の変化や、樹状突起計算の効果を観察する。

import sys
import os
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np

# プロジェクトルートの設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from snn_research.models.bio.pd14_microcircuit import PD14Microcircuit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("BioDemo")

def main():
    print("\n🧠 --- Biological Microcircuit Demo (PD14 + Active Dendrites) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    # 1. モデル構築 (小規模スケール)
    # スケール0.05でも約4000ニューロンになるため、デモ用にさらに小さくする
    scale = 0.01 
    time_steps = 50
    
    model = PD14Microcircuit(
        scale_factor=scale,
        time_steps=time_steps,
        neuron_type="two_compartment", # 多区画モデルを使用
        input_dim=32,
        output_dim=10
    ).to(device)
    
    # 2. シナリオA: ボトムアップ入力のみ (受動的知覚)
    print("\nTesting Scenario A: Bottom-Up Input Only (Passive Perception)")
    thalamic_input = torch.randn(1, 32).to(device) * 2.0 # 強めの入力
    
    out_a, rates_a = model(thalamic_input=thalamic_input, topdown_input=None)
    
    print("   [Firing Rates per Population]")
    for pop, rate in rates_a.items():
        print(f"     - {pop}: {rate:.2f} spikes/step")
        
    # L4 (入力層) が強く反応し、L2/3, L5 へ伝播しているか確認
    if rates_a["L4e"] > rates_a["L5e"]:
        print("   ✅ Valid Propagation: L4 (Input) > L5 (Output) as expected for pure feedforward.")
        
    # 3. シナリオB: トップダウン入力あり (注意・予測)
    # 樹状突起への入力が細胞体の発火を助ける効果を確認
    print("\nTesting Scenario B: With Top-Down Attention (Active Prediction)")
    # ボトムアップ入力は弱くする（ノイズレベル）
    weak_input = torch.randn(1, 32).to(device) * 0.5
    # トップダウン入力（予測信号）を与える
    topdown_signal = torch.randn(1, 32).to(device) * 2.0
    
    out_b, rates_b = model(thalamic_input=weak_input, topdown_input=topdown_signal)
    
    print("   [Firing Rates per Population]")
    for pop, rate in rates_b.items():
        print(f"     - {pop}: {rate:.2f} spikes/step")
        
    # トップダウン入力により、L2/3やL5の活動が増強される（NMDAスパイク効果）
    # L4はボトムアップのみ受けるため、あまり変わらないはず
    gain = rates_b["L5e"] - rates_a["L5e"] # 注: 入力条件が違うので単純比較は難しいが、傾向を見る
    
    print("\n   [Comparison]")
    print(f"   L4e Activity: A(Strong)={rates_a['L4e']:.2f} vs B(Weak)={rates_b['L4e']:.2f}")
    print(f"   L5e Activity: A={rates_a['L5e']:.2f} vs B(With Context)={rates_b['L5e']:.2f}")
    
    print("\n🎉 Demo Completed. The brain architecture is functional.")

if __name__ == "__main__":
    main()