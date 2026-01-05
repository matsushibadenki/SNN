# ファイルパス: scripts/run_phase7_os_simulation.py
# Title: Phase 7 Brain OS Simulation (Config Fix)
# Description:
#   NeuromorphicScheduler を使用して、複数の認知モジュール（視覚、言語、制御）が
#   限られたエネルギーリソースを巡って競合する様子をシミュレーションする。
#   修正: BrainContainer初期化時に設定ファイル（モデル定義）をロードするように修正し、
#   SNNCoreの初期化エラーを解消。

from snn_research.cognitive_architecture.neuromorphic_scheduler import NeuromorphicScheduler, BrainProcess, ProcessBid
from app.containers import BrainContainer  # E402 fixed
import sys
import os
import logging
import time
from typing import Dict, Any

# プロジェクトルート設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ロギング設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger("BrainOS")

# --- 入札戦略関数 (Bid Strategies) ---


def bid_visual(module, input_data: Any, context: Dict[str, Any]) -> ProcessBid:
    """視覚プロセスの入札戦略"""
    # 画像入力なら優先度高
    if isinstance(input_data, dict) and input_data.get("type") == "image":
        priority = 0.9
        cost = 15.0
        intent = "Process visual input"
    else:
        # 画像以外でも、意識に視覚的要素があれば少し反応
        if context.get("consciousness") and "visual" in str(context["consciousness"]):
            priority = 0.4
            cost = 5.0
            intent = "Imagine visual scene"
        else:
            priority = 0.0
            cost = 0.0
            intent = "Idle"

    return ProcessBid("VisualCortex", priority, cost, intent)


def bid_language(module, input_data: Any, context: Dict[str, Any]) -> ProcessBid:
    """言語思考プロセスの入札戦略"""
    # テキスト入力なら反応
    if isinstance(input_data, dict) and input_data.get("type") == "text":
        priority = 0.8
        cost = 10.0
        intent = "Process text input"
    # 何もなくても思考しようとする（デフォルトモードネットワーク）
    else:
        # エネルギーが十分にあれば思考する
        if context["energy"] > 300:
            priority = 0.3
            cost = 8.0
            intent = "Daydream / Think"
        else:
            priority = 0.1
            cost = 2.0
            intent = "Minimal thought"

    return ProcessBid("ThinkingEngine", priority, cost, intent)


def bid_amygdala(module, input_data: Any, context: Dict[str, Any]) -> ProcessBid:
    """扁桃体（情動）の入札戦略"""
    # 常に一定の監視を行う（生存本能）
    priority = 0.5
    cost = 2.0
    intent = "Monitor emotion"

    # テキストに危険な単語があれば優先度急上昇
    if isinstance(input_data, dict) and input_data.get("type") == "text":
        text = input_data.get("content", "")
        if "danger" in text or "error" in text:
            priority = 1.0  # 最優先（割り込み）
            intent = "DANGER RESPONSE"

    return ProcessBid("Amygdala", priority, cost, intent)

# --- 実行関数 (Executors) ---


def exec_visual(module, input_data):
    # VisualCortexのダミー実行
    # 本来は module.perceive_and_upload(...) を呼ぶ
    return {"status": "seen", "upload_content": {"type": "visual", "features": "tensor..."}}


def exec_language(module, input_data):
    # ThinkingEngineのダミー実行
    time.sleep(0.1)  # 思考時間をシミュレート
    return {"status": "thought", "upload_content": "I think therefore I am."}


def exec_amygdala(module, input_data):
    return {"status": "felt", "upload_content": {"type": "emotion", "valence": 0.1}}


def main():
    print("\n" + "="*60)
    print("🧠 SNN Phase 7: Brain OS Simulation")
    print("   Multi-Agent Competition & Resource Arbitration")
    print("="*60)

    # 1. コンテナからコンポーネント取得
    container = BrainContainer()

    # --- 修正: 設定ファイルのロード (ThinkingEngineの初期化に必要) ---
    base_config_path = "configs/templates/base_config.yaml"
    model_config_path = "configs/models/small.yaml"

    if os.path.exists(base_config_path):
        container.config.from_yaml(base_config_path)

    if os.path.exists(model_config_path):
        container.config.from_yaml(model_config_path)
    else:
        # 設定ファイルがない場合のフォールバック
        print("⚠️ Model config not found, using fallback configuration.")
        container.config.from_dict({
            "model": {
                "architecture_type": "predictive_coding",
                "d_model": 64,
                "d_state": 32,
                "num_layers": 2,
                "time_steps": 16,
                "neuron": {"type": "lif"}
            },
            "data": {"tokenizer_name": "gpt2"}
        })
    # ---------------------------------------------------------

    astrocyte = container.astrocyte_network()
    workspace = container.global_workspace()

    # 2. OSスケジューラの初期化
    os_kernel = NeuromorphicScheduler(astrocyte, workspace)

    # 3. プロセスの登録 (BrainProcessとしてラップ)

    print("   - Initializing Brain Processes...")

    # コンポーネントの取得 (設定ロード後に実行)
    visual_cortex = container.visual_cortex()
    thinking_engine = container.thinking_engine()
    amygdala = container.amygdala()

    proc_visual = BrainProcess(
        "VisualCortex", visual_cortex, bid_visual, exec_visual)
    os_kernel.register_process(proc_visual)

    proc_lang = BrainProcess(
        "ThinkingEngine", thinking_engine, bid_language, exec_language)
    os_kernel.register_process(proc_lang)

    proc_amygdala = BrainProcess(
        "Amygdala", amygdala, bid_amygdala, exec_amygdala)
    os_kernel.register_process(proc_amygdala)

    # 4. シミュレーション実行

    # Scenario A: 通常状態 (テキスト入力)
    print("\n--- Scenario A: Routine Text Processing ---")
    input_a = {"type": "text", "content": "Hello world."}
    os_kernel.step(input_a)

    # Scenario B: 緊急事態 (危険な入力 + リソース競合)
    print("\n--- Scenario B: Emergency Interrupt ---")
    input_b = {"type": "text", "content": "System error! danger!"}
    os_kernel.step(input_b)

    # Scenario C: 視覚入力 (高コスト)
    print("\n--- Scenario C: High-Cost Visual Processing ---")
    input_c = {"type": "image", "content": "image_data..."}
    os_kernel.step(input_c)

    # Scenario D: エネルギー枯渇 (Starvation)
    print("\n--- Scenario D: Energy Starvation ---")
    # エネルギーを強制的に下げる
    astrocyte.current_energy = 20.0
    print(f"   📉 Energy dropped to {astrocyte.current_energy}!")

    # 低エネルギー下で複合入力
    input_d = {"type": "text", "content": "Think deeply about the universe."}
    # 言語思考（高コスト・中優先度）は却下され、扁桃体（低コスト・中優先度）のみ動く可能性がある
    os_kernel.step(input_d)

    print("\n✅ OS Simulation Complete.")


if __name__ == "__main__":
    main()
