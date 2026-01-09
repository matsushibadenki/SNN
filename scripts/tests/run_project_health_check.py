# ファイルパス: scripts/tests/run_project_health_check.py
# Title: SNN Project Health Check v5.6 (Fix SFormer & BN)
# Description:
#   プロジェクト全体の健全性を検証する統合チェックスクリプト。
#   v5.6: Spikformerの引数修正とHybridモデルのバッチサイズ調整(BatchNorm対応)。

import sys
import os
import time
import subprocess
import logging

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
import torch.nn as nn
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
    logger.info("🩺 SNNプロジェクト ヘルスチェック v5.6 (Fix SFormer & BN) 開始")
    print("-" * 60)

    checks = []

    # 1. Unit Tests (Pytest)
    checks.append(run_command(
        f"{sys.executable} -m pytest tests/ -x -q --disable-warnings", "Unit Tests: Pytest Suite (Quick)"))

    # 2. Core: SNNCore & SFormer Init
    # Fix: architecture_typeを指定
    # Fix: Spikformerの引数を修正 (input_dim -> embed_dim, img_size指定)
    checks.append(check_python_api("Core: SNNCore & SFormer Init", """
from snn_research.core.snn_core import SNNCore
from snn_research.models.transformer.spikformer import Spikformer

# SNNCoreには architecture_type が必須
config = {
    'architecture_type': 'spiking_transformer',
    'd_model': 64,
    'num_layers': 2,
    'time_steps': 4,
    'neuron': {'type': 'lif'}
}
model = SNNCore(config=config, vocab_size=100)

# Spikformerのテスト
# img_sizeなどを明示し、embed_dimを使用
sformer = Spikformer(img_size_h=32, img_size_w=32, embed_dim=64, num_classes=10)
print("Models initialized successfully")
"""))

    # 3. Core: BitSpikeMamba
    # Fix: 足りない引数を追加
    checks.append(check_python_api("Core: BitSpikeMamba (1.58bit LLM)", """
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba

model = BitSpikeMamba(
    vocab_size=100,
    d_model=32,
    d_state=16,
    d_conv=4,
    expand=2,
    num_layers=1,
    time_steps=4,
    neuron_config={'type': 'lif', 'tau_mem': 0.5, 'v_reset': 0.0}
)
x = torch.randint(0, 100, (1, 10))
y = model(x)
print("Forward pass successful")
"""))

    # 4. Cognitive: ArtificialBrain Cycle
    checks.append(check_python_api("Cognitive: ArtificialBrain Cycle", """
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
brain = ArtificialBrain(config={'stm_capacity': 10})
res = brain.process_step(sensory_input="test_input")
print(f"Cycle result: {res}")
"""))

    # 5. Cognitive: Sleep & Consolidation Demo
    checks.append(run_command(
        f"{sys.executable} scripts/demos/learning/run_sleep_cycle_demo.py", "Cognitive: Sleep & Consolidation Demo"))

    # 6. Agent: Planner SNN
    # Fix: 足りない引数を追加
    checks.append(check_python_api("Agent: Planner SNN (Reasoning)", """
from snn_research.cognitive_architecture.planner_snn import PlannerSNN

planner = PlannerSNN(
    vocab_size=50,
    d_model=32,
    d_state=16,
    num_layers=1,
    time_steps=4,
    n_head=2,
    num_skills=5,
    neuron_config={'type': 'lif'}
)
print("Planner initialized")
"""))

    # 7. Logic: LogicGatedSNN
    # Fix: 引数名変更
    checks.append(check_python_api("Logic: LogicGatedSNN (Neuro-Symbolic)", """
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN

layer = LogicGatedSNN(in_features=10, out_features=5)
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

    # 10. Evolution: Self-Evolving Agent Master
    # Fix: 依存オブジェクトをMockで注入
    checks.append(check_python_api("Evolution: Self-Evolving Agent Master", """
from snn_research.agent.self_evolving_agent import SelfEvolvingAgentMaster

# Mock dependencies
class MockObj:
    def __init__(self, *args, **kwargs): pass
    def to(self, device): return self

mock_planner = MockObj()
mock_registry = MockObj()
mock_memory = MockObj()
mock_crawler = MockObj()
mock_meta = MockObj()
mock_motivation = MockObj()

agent = SelfEvolvingAgentMaster(
    name="TestEvolver",
    planner=mock_planner,
    model_registry=mock_registry,
    memory=mock_memory,
    web_crawler=mock_crawler,
    meta_cognitive_snn=mock_meta,
    motivation_system=mock_motivation,
    model_config_path=None
)
print("Agent Master initialized")
"""))

    # 11. Model: Hybrid CNN-SNN
    # Fix: クラス名修正
    # Fix: バッチサイズを2に変更 (BatchNormエラー回避)
    checks.append(check_python_api("Model: Hybrid CNN-SNN (Vision)", """
from snn_research.models.cnn.hybrid_cnn_snn_model import HybridCnnSnnModel

ann_config = {'name': 'mobilenet_v2', 'output_features': 1280, 'pretrained': False}
snn_config = {'d_model': 32, 'n_head': 2, 'num_layers': 1}
neuron_config = {'type': 'lif'}

model = HybridCnnSnnModel(
    vocab_size=10, 
    time_steps=4,
    ann_frontend=ann_config,
    snn_backend=snn_config,
    neuron_config=neuron_config
)
# Batch size > 1 required for BatchNorm training mode
x = torch.randn(2, 3, 32, 32)
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
