# ファイルパス: scripts/tests/run_project_health_check.py
# Title: SNN Project Health Check v5.4
# Description:
#   プロジェクト全体の健全性を検証する統合チェックスクリプト。
#   ArtificialBrainのメソッド名変更(run_cognitive_cycle -> process_step)に対応。

import sys
import os
import time
import subprocess
import logging
import torch
import traceback

# パス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ロガー設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HealthCheck")


def run_command(command, description):
    """サブプロセスでコマンドを実行し、結果をログ出力する"""
    logger.info(f"Checking {description}...")
    start_time = time.time()

    try:
        # タイムアウトを設定して無限ループ防止
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✅ 成功: {description} ({elapsed:.2f}s)")
            return True
        else:
            logger.error(f"❌ 失敗: {description}")
            print(f"[STDERR]:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"⏰ タイムアウト: {description} (300s超過)")
        return False
    except Exception as e:
        logger.error(f"⚠️ エラー: {description} - {str(e)}")
        return False


def check_python_api(description, code_snippet):
    """Pythonコードスニペットを実行して検証"""
    logger.info(f"Checking {description}...")
    start_time = time.time()

    try:
        # 必要なインポートを含めたラッパー
        full_code = f"""
import sys
import os
import torch
sys.path.append(os.getcwd())
{code_snippet}
"""
        # サブプロセスで実行（環境汚染を防ぐため）
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=60
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✅ 成功: {description} ({elapsed:.2f}s)")
            return True
        else:
            logger.error(f"❌ 失敗: {description}")
            print(f"[STDERR]:\n{result.stderr}")
            return False

    except Exception as e:
        logger.error(f"⚠️ 実行エラー: {description} - {str(e)}")
        return False


def main():
    logger.info("🩺 SNNプロジェクト ヘルスチェック v5.4 (Pytest Integrated) 開始")
    print("-" * 60)

    checks = []

    # 1. Unit Tests (Pytest)
    # 全テストではなく、高速なコア機能のみを対象とするマーカー等を推奨するが、
    # ここでは既存の構成に従い tests/ を実行（-x で初回失敗時に停止）
    checks.append(run_command(
        f"{sys.executable} -m pytest tests/ -x -q --disable-warnings", "Unit Tests: Pytest Suite (Quick)"))

    # 2. Core: SNNCore & SFormer Init
    checks.append(check_python_api("Core: SNNCore & SFormer Init", """
from snn_research.core.snn_core import SNNCore
from snn_research.models.transformer.spikformer import Spikformer

model = SNNCore(config={'d_model': 64, 'num_layers': 2}, vocab_size=100)
sformer = Spikformer(input_dim=64, num_classes=10)
print("Models initialized successfully")
"""))

    # 3. Core: BitSpikeMamba
    checks.append(check_python_api("Core: BitSpikeMamba (1.58bit LLM)", """
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
model = BitSpikeMamba(vocab_size=100, d_model=32)
x = torch.randint(0, 100, (1, 10))
y = model(x)
print("Forward pass successful")
"""))

    # 4. Cognitive: ArtificialBrain Cycle (Fix: run_cognitive_cycle -> process_step)
    checks.append(check_python_api("Cognitive: ArtificialBrain Cycle", """
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
brain = ArtificialBrain(config={'stm_capacity': 10})
# [Fix] Updated method name
res = brain.process_step(sensory_input="test_input")
print(f"Cycle result: {res}")
"""))

    # 5. Cognitive: Sleep & Consolidation Demo
    checks.append(run_command(
        f"{sys.executable} scripts/demos/learning/run_sleep_cycle_demo.py", "Cognitive: Sleep & Consolidation Demo"))

    # 6. Agent: Planner SNN
    checks.append(check_python_api("Agent: Planner SNN (Reasoning)", """
from snn_research.cognitive_architecture.planner_snn import PlannerSNN
planner = PlannerSNN(vocab_size=50, num_skills=5)
print("Planner initialized")
"""))

    # 7. Logic: LogicGatedSNN
    checks.append(check_python_api("Logic: LogicGatedSNN (Neuro-Symbolic)", """
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN
layer = LogicGatedSNN(input_dim=10, output_dim=5)
x = torch.randn(1, 10)
y = layer(x)
print("Logic gate forward pass successful")
"""))

    # 8. IO: UniversalSpikeEncoder
    checks.append(check_python_api("IO: UniversalSpikeEncoder (Image/Audio/Text)", """
from snn_research.io.universal_encoder import UniversalSpikeEncoder
encoder = UniversalSpikeEncoder()
img = torch.randn(1, 3, 224, 224)
spikes = encoder.encode(img, modality='image')
print("Encoding successful")
"""))

    # 9. Distillation: Manager
    checks.append(run_command(
        f"{sys.executable} scripts/demos/learning/run_distillation_demo.py", "Distill: Knowledge Distillation Manager"))

    # 10. Evolution: Self-Evolving Agent
    checks.append(check_python_api("Evolution: Self-Evolving Agent Master", """
from snn_research.agent.self_evolving_agent import SelfEvolvingAgentMaster
agent = SelfEvolvingAgentMaster(name="TestEvolver")
print("Agent Master initialized")
"""))

    # 11. Model: Hybrid CNN-SNN
    checks.append(check_python_api("Model: Hybrid CNN-SNN (Vision)", """
from snn_research.models.cnn.hybrid_cnn_snn_model import HybridCNNSNN
model = HybridCNNSNN(num_classes=10)
x = torch.randn(1, 3, 32, 32)
y = model(x)
print("Hybrid model forward pass successful")
"""))

    # 12. Application: Industrial Eye (DVS)
    checks.append(run_command(
        f"{sys.executable} scripts/demos/visual/run_industrial_eye_demo.py", "App: Industrial Eye (DVS Processing)"))

    # 13. Application: ECG Analysis
    checks.append(run_command(
        f"{sys.executable} scripts/experiments/applications/run_ecg_analysis.py", "App: ECG Analysis (Temporal)"))

    # 14. Training: Smoke Test
    checks.append(run_command(
        f"{sys.executable} scripts/training/trainers/train_overfit_demo.py", "Train: Overfit Smoke Test"))

    # 15. Hardware: Compiler
    checks.append(run_command(
        f"{sys.executable} scripts/tests/run_compiler_test.py", "Hardware: Neuromorphic Compiler"))

    # 集計
    passed = sum(checks)
    total = len(checks)

    print("-" * 60)
    logger.info(f"📊 最終結果: {passed} / {total} 項目合格")

    if passed == total:
        logger.info("✨ 全てのヘルスチェックに合格しました！プロジェクトは健全です。")
        sys.exit(0)
    else:
        logger.error(f"⚠️ {total - passed} 個のコンポーネントで問題が発生しています。ログを確認してください。")
        sys.exit(1)


if __name__ == "__main__":
    main()
