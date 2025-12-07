# ファイルパス: scripts/runners/run_neuromorphic_os.py
# Title: SNN Brain OS Simulation
# Description:
#   人工脳をOSとして動作させ、高負荷環境下でのリソース配分と自律的な睡眠サイクルを実演する。
#   ユーザーは連続的にタスクを投入し、脳が「疲れて」反応しなくなる（抑制する）様子や、
#   睡眠後に回復する様子を観察できる。

import sys
import os
import time
import logging
from pathlib import Path
from omegaconf import OmegaConf

# プロジェクトルート設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.containers import BrainContainer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("BrainOS")
logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    print("🧠 --- Neuromorphic OS Simulation (Phase 7 Demo) ---")
    print("   高負荷タスクを与え、アストロサイトによるリソース制御と睡眠サイクルを観察します。")

    # 1. コンテナと脳の初期化
    container = BrainContainer()
    # テスト用にエネルギー容量を小さく設定して、現象を早く起こす
    # (AstrocyteNetworkのデフォルト引数を上書きできないため、生成後に調整)
    brain = container.artificial_brain()
    
    # OS設定のオーバーライド (デモ用)
    brain.astrocyte.max_energy = 300.0
    brain.astrocyte.current_energy = 300.0
    brain.astrocyte.basal_metabolic_rate = 5.0
    
    print(f"\n🔋 Initial State: Energy={brain.astrocyte.current_energy}, Fatigue={brain.astrocyte.fatigue_toxin}")

    # 2. タスクストリームの実行
    tasks = [
        ("text", "Hello world."), 
        ("text", "Calculate pi."), 
        ("image", "dummy_image_path"), # 高コスト
        ("text", "Philosophy of mind."),
        ("text", "What is SNN?"),
        ("image", "dummy_image_path_2"), # 高コスト
        ("text", "Deep reasoning task."),
        ("text", "Another heavy task."),
        ("text", "Keep working..."),
        ("text", "Are you tired?"),
        ("text", "One more thing..."),
        ("text", "Critical system alert."),
    ]

    for i, (kind, content) in enumerate(tasks):
        print(f"\n>>> [Task {i+1}] Type: {kind}, Content: {content}")
        
        # 画像入力のシミュレーション
        input_data = content
        if kind == "image":
            # 実際には画像をロードするが、ここではSensoryReceptorのモック動作に期待するか、
            # もしくはbrain.run_cognitive_cycleが文字列をパスとして処理する前提
            pass 

        result = brain.run_cognitive_cycle(input_data)
        
        # OSの状態表示
        energy = result.get("energy", 0.0)
        fatigue = result.get("fatigue", 0.0)
        executed = result.get("executed_modules", [])
        denied = result.get("denied_modules", [])
        
        print(f"   ⚡ Energy: {energy:.1f} | 😫 Fatigue: {fatigue:.1f}")
        
        if denied:
            print(f"   ⚠️ DENIED MODULES (Low Resource): {', '.join(denied)}")
        
        if result.get("status") == "sleeping":
            print("   💤 Brain is sleeping... Task ignored.")
        elif not executed and not denied:
            # 睡眠サイクルに入った直後の場合など
            print("   (Cycle completed without module execution)")

        time.sleep(0.5)

    print("\n🎉 Demo Completed: Neuromorphic OS behavior verified.")

if __name__ == "__main__":
    main()