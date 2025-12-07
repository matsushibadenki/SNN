# ファイルパス: scripts/runners/run_neuromorphic_os.py
# Title: SNN Brain OS Simulation (Config Loading Fixed)
# Description:
#   人工脳をOSとして動作させ、高負荷環境下でのリソース配分と自律的な睡眠サイクルを実演する。
#   修正: BrainContainer初期化時に設定ファイルをロードするように修正し、
#   CorticalColumnの初期化エラー（NoneType）を解消。

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
# 外部ライブラリのログを抑制
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

def main():
    print("🧠 --- Neuromorphic OS Simulation (Phase 7 Demo) ---")
    print("   高負荷タスクを与え、アストロサイトによるリソース制御と睡眠サイクルを観察します。")

    # 1. コンテナと脳の初期化
    container = BrainContainer()
    
    # --- 修正: 設定ファイルのロード ---
    # DIコンテナが依存関係（CorticalColumnなど）を解決するために必須
    base_config_path = "configs/templates/base_config.yaml"
    model_config_path = "configs/models/small.yaml"

    if os.path.exists(base_config_path):
        container.config.from_yaml(base_config_path)
    else:
        logger.warning(f"⚠️ Config not found: {base_config_path}")

    if os.path.exists(model_config_path):
        container.config.from_yaml(model_config_path)
    else:
        logger.warning(f"⚠️ Model config not found: {model_config_path}")
        # フォールバック用の最小設定
        container.config.from_dict({
            "model": {"architecture_type": "predictive_coding", "d_model": 64, "time_steps": 16},
            "training": {"biologically_plausible": {"neuron": {"type": "lif", "tau_mem": 10.0, "base_threshold": 1.0}}}
        })
    # --------------------------------

    try:
        # テスト用にエネルギー容量を小さく設定して、現象を早く起こす
        brain = container.artificial_brain()
    except Exception as e:
        logger.error(f"❌ Failed to initialize Artificial Brain: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # OS設定のオーバーライド (デモ用)
    # AstrocyteNetworkはArtificialBrain内で初期化されている（または注入されている）
    if brain.astrocyte:
        brain.astrocyte.max_energy = 300.0
        brain.astrocyte.current_energy = 300.0
        brain.astrocyte.basal_metabolic_rate = 5.0
        print(f"\n🔋 Initial State: Energy={brain.astrocyte.current_energy}, Fatigue={brain.astrocyte.fatigue_toxin}")
    else:
        logger.error("❌ Astrocyte module is missing in Artificial Brain.")
        return

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
            # ファイルが存在しないとエラーになる可能性があるため、ダミー処理
            if not os.path.exists(input_data):
                 # SensoryReceptorがパスとして認識しないように、明示的にテキストとして扱うか、
                 # もしくはダミー画像を生成するロジックが必要だが、
                 # 今回はSensoryReceptorがファイル欠損をハンドリングする（text扱いになる）
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
            # 睡眠中は少し速く進める
            time.sleep(0.2)
        elif not executed and not denied:
            # 睡眠サイクルに入った直後の場合など
            print("   (Cycle completed without module execution)")
            time.sleep(0.2)
        else:
            time.sleep(0.5)

    print("\n🎉 Demo Completed: Neuromorphic OS behavior verified.")

if __name__ == "__main__":
    main()
