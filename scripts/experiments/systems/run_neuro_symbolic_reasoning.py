# ファイルパス: scripts/experiments/systems/run_neuro_symbolic_reasoning.py
# 日本語タイトル: Neuro-Symbolic Reasoning Simulation
# 目的: SNNによる「知覚」と、ルールエンジンによる「論理」を組み合わせた
#       ハイブリッドな意思決定プロセスを実証する。

from snn_research.cognitive_architecture.neuro_symbolic_bridge import NeuroSymbolicBridge, SimpleLogicEngine
from snn_research.models.experimental.brain_v4 import SynestheticBrain
import os
import sys
import torch
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())


# ロギング設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NeuroSymbolicSim")


def main():
    logger.info("🧠 Starting Neuro-Symbolic Reasoning Experiment...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Define Knowledge Base (Concepts & Rules) ---
    # コンセプト定義
    concepts = [
        "RED_LIGHT",    # 知覚: 赤信号
        "GREEN_LIGHT",  # 知覚: 青信号
        "PEDESTRIAN",   # 知覚: 歩行者あり
        "CLEAR_ROAD",   # 知覚: 道が空いている
        "STOP",         # 行動概念: 止まれ
        "GO",           # 行動概念: 進め
        "DANGER"        # 抽象概念: 危険
    ]

    # ルール定義 (If A [and B] Then C)
    # タプル形式: (条件1, 条件2orNone, 結果)
    rules = [
        ("RED_LIGHT", None, "STOP"),                # 赤なら止まれ
        ("PEDESTRIAN", None, "STOP"),               # 人がいたら止まれ
        ("GREEN_LIGHT", "CLEAR_ROAD", "GO"),        # 青かつ道が空いていれば進め
        ("STOP", None, "DANGER_AVOIDANCE")          # (追加ルール) 止まることは危険回避行動
    ]

    logic_engine = SimpleLogicEngine(rules)
    logger.info(
        f"   - Defined {len(concepts)} concepts and {len(rules)} logic rules.")

    # --- 2. Initialize Models ---
    # A. SNN Brain (Perception)
    # 画像入力(64次元)を受け取り、スパイク特徴量(32次元)を出力する脳
    _ = SynestheticBrain(
        vocab_size=10, d_model=32, num_layers=1, time_steps=4, device=device
    )

    # B. Neuro-Symbolic Bridge
    # Brainの出力(d_model=32)をシンボルに変換し、逆にシンボルをBrainの入力(32)に戻す
    ns_bridge = NeuroSymbolicBridge(
        input_dim=32,
        embed_dim=32,
        concepts=concepts
    ).to(device)

    # --- 3. Simulation: Traffic Scenario ---
    logger.info("\n--- Scenario: Approaching an Intersection ---")

    # ケース1: 赤信号が見えている状況
    # 実際は画像データだが、ここでは「赤信号の特徴」を持ったランダムベクトルで代用
    # (本来は学習によって、特定の画像パターンが特定のConceptニューロンを発火させるようになる)

    # 疑似的な学習済み状態を作るため、Bridgeの重みを強制的に設定 (デモ用ハック)
    # RED_LIGHT(idx=0) に対応する入力次元(dim=0)の重みを高くする
    with torch.no_grad():
        ns_bridge.extractor.weight.fill_(-1.0)  # デフォルトは抑制
        # 入力[0]が高ければ RED_LIGHT[0] が発火
        ns_bridge.extractor.weight[0, 0] = 2.0
        ns_bridge.extractor.bias.fill_(-0.5)

        # 入力[1]が高ければ PEDESTRIAN[2] が発火
        ns_bridge.extractor.weight[2, 1] = 2.0

    # 入力データ生成: 次元0の値が高い -> 赤信号の特徴
    visual_input = torch.zeros(1, 32, device=device)
    visual_input[0, 0] = 5.0  # 強い赤信号刺激

    logger.info("👀 Step 1: Perception (Neural Processing)")
    # Brainで処理（ここでは簡略化のため特徴量を直接Bridgeへ）
    # 実際: image -> brain.encoder -> features -> ns_bridge
    neural_features = visual_input

    # 1. 記号接地 (Grounding)
    detected_symbols = ns_bridge.extract_symbols(
        neural_features, threshold=0.5)
    detected_names = [s.name for s in detected_symbols]
    logger.info(f"   -> Detected Concepts: {detected_names}")

    # 2. 論理推論 (Reasoning)
    logger.info("⚙️ Step 2: Logical Reasoning (Symbolic Processing)")
    inferred_facts = logic_engine.infer(detected_names)
    logger.info(f"   -> Inferred Facts: {inferred_facts}")

    # 3. 運動制御へのフィードバック (Action)
    logger.info("⚡ Step 3: Action Generation (Neural Injection)")
    # 推論された事実の中に「行動(STOP/GO)」が含まれているか確認
    decision = "WAIT"
    if "STOP" in inferred_facts:
        decision = "BRAKE"
    elif "GO" in inferred_facts:
        decision = "ACCELERATE"

    logger.info(f"   -> Final Decision: {decision}")

    # 推論結果を脳に戻す (Optional: メタ認知)
    # これにより脳は「なぜ止まるのか(赤信号だから)」という文脈を保持できる
    feedback_signal = ns_bridge.inject_symbols(inferred_facts)
    logger.info(
        f"   -> Feedback Signal Strength: {feedback_signal.norm().item():.4f}")

    # --- Scenario 2: Green Light but Pedestrian ---
    logger.info("\n--- Scenario 2: Green Light but Pedestrian Crossing ---")
    # 入力: 青信号(仮にdim=3) と 歩行者(dim=1)
    with torch.no_grad():
        ns_bridge.extractor.weight[1, 3] = 2.0  # GREEN_LIGHT

    visual_input_2 = torch.zeros(1, 32, device=device)
    visual_input_2[0, 3] = 5.0  # Green Light
    visual_input_2[0, 1] = 5.0  # Pedestrian

    # 実行
    syms_2 = ns_bridge.extract_symbols(visual_input_2)
    names_2 = [s.name for s in syms_2]
    logger.info(f"   -> Perception: {names_2}")

    facts_2 = logic_engine.infer(names_2)
    logger.info(f"   -> Reasoning: {facts_2}")

    if "STOP" in facts_2 and "GO" in facts_2:
        # 矛盾(Conflict)の解決: ルールの優先順位や安全係数が必要
        logger.warning("   -> Conflict Detected! (STOP and GO)")
        final_dec = "EMERGENCY_STOP"  # 安全側に倒す
    elif "STOP" in facts_2:
        final_dec = "BRAKE (Pedestrian Priority)"
    else:
        final_dec = "GO"

    logger.info(f"   -> Final Decision: {final_dec}")

    logger.info("\n✅ Neuro-Symbolic Demo Completed.")


if __name__ == "__main__":
    main()
