# ファイルパス: scripts/tests/run_project_health_check.py
# 日本語タイトル: SNNプロジェクト 全機能網羅ヘルスチェック v5.2 (Path Fixed)
# 概要: エラーが発生していたファイルパス（runners -> demos, scripts直下 -> experiments等）を
#       実際のディレクトリ構成に合わせて修正しました。

import os
import sys
import subprocess
import time
import logging
from typing import List, Tuple

project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HealthCheck")


def run_command(command: str, description: str) -> Tuple[bool, float]:
    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + \
        (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    try:
        # タイムアウトを90秒に設定
        result = subprocess.run(
            command, shell=True, env=env, capture_output=True, text=True, timeout=90)
        duration = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✅ 成功: {description} ({duration:.2f}s)")
            return True, duration
        else:
            logger.error(
                f"❌ 失敗: {description}\n[STDERR]:\n{result.stderr.strip()}")
            return False, duration
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ タイムアウト: {description} (90s超過)")
        return False, 90.0
    except Exception as e:
        logger.error(f"⚠️ 例外: {description} ({e})")
        return False, 0.0


def main():
    logger.info("🩺 SNNプロジェクト ヘルスチェック v5.2 (Path Fixed) 開始")
    python_cmd = sys.executable
    if " " in python_cmd:
        python_cmd = f'"{python_cmd}"'

    checks = [
        # --- 1. Core Architecture ---
        {
            "name": "Core: SNNCore & SFormer Init",
            "cmd": f"{python_cmd} -c \"from snn_research.core.snn_core import SNNCore; import torch; model=SNNCore({{'architecture_type':'sformer'}}, 100); model(torch.randint(0,100,(1,16)))\""
        },
        {
            "name": "Core: BitSpikeMamba (1.58bit LLM)",
            "cmd": f"{python_cmd} -c \"from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba; import torch; m=BitSpikeMamba(vocab_size=100, d_model=64, d_state=16, d_conv=4, expand=2, num_layers=1, time_steps=4, neuron_config={{'type':'lif'}}); m(torch.randint(0,100,(1,4)))\""
        },

        # --- 2. Cognitive System ---
        {
            "name": "Cognitive: ArtificialBrain Cycle",
            "cmd": f"{python_cmd} -c \"from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain; import torch; brain=ArtificialBrain(); brain.run_cognitive_cycle(torch.randn(1,3,32,32))\""
        },
        {
            "name": "Cognitive: Sleep & Consolidation Demo",
            # [修正] scripts/demos/ -> scripts/demos/learning/
            "cmd": f"{python_cmd} scripts/demos/learning/run_sleep_cycle_demo.py"
        },

        # --- 3. Agent & Logic ---
        {
            "name": "Agent: Planner SNN (Reasoning)",
            "cmd": f"{python_cmd} -c \"from snn_research.cognitive_architecture.planner_snn import PlannerSNN; import torch; p=PlannerSNN(vocab_size=100, d_model=64, d_state=16, num_layers=1, time_steps=4, n_head=2, num_skills=5, neuron_config={{'type':'lif'}}); p(torch.randint(0,100,(1,4)))\""
        },
        {
            "name": "Logic: LogicGatedSNN (Neuro-Symbolic)",
            "cmd": f"{python_cmd} -c \"from snn_research.core.layers.logic_gated_snn import LogicGatedSNN; import torch; l=LogicGatedSNN(64, 64); l(torch.randn(10,64))\""
        },

        # --- 4. IO & Multimodal ---
        {
            "name": "IO: UniversalSpikeEncoder (Image/Audio/Text)",
            "cmd": f"{python_cmd} -c \"from snn_research.io.universal_encoder import UniversalSpikeEncoder; import torch; enc=UniversalSpikeEncoder(); enc.encode(torch.randn(1,3,32,32), 'image'); enc.encode_text_str('hello')\""
        },

        # --- 5. Distillation & Evolution ---
        {
            "name": "Distill: Knowledge Distillation Manager",
            "cmd": f"{python_cmd} -c \"from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager; import torch; from omegaconf import OmegaConf; conf=OmegaConf.create({{'model':{{'time_steps':16}}, 'training':{{'batch_size':2, 'epochs':1, 'log_dir':'workspace/logs'}}}}); KnowledgeDistillationManager(student_model=torch.nn.Linear(10,10), trainer=None, model_registry=None, device='cpu', config=conf, teacher_model_name='gpt2')\""
        },
        {
            "name": "Evolution: Self-Evolving Agent Master",
            "cmd": f"{python_cmd} -c \"from snn_research.agent.self_evolving_agent import SelfEvolvingAgentMaster; agent=SelfEvolvingAgentMaster('test_agent', planner=None, model_registry=None, memory=None, web_crawler=None, meta_cognitive_snn=None, motivation_system=None)\""
        },

        # --- 6. Applications & Hybrid Models ---
        {
            "name": "Model: Hybrid CNN-SNN (Vision)",
            "cmd": f"{python_cmd} -c \"from snn_research.models.cnn.hybrid_cnn_snn_model import HybridCnnSnnModel; import torch; m=HybridCnnSnnModel(vocab_size=10, time_steps=4, ann_frontend={{'name':'mobilenet_v2', 'output_features':1280}}, snn_backend={{'d_model':64, 'n_head':4, 'num_layers':2}}, neuron_config={{'type':'lif'}}); m(torch.randn(1,3,224,224))\""
        },
        {
            "name": "App: Industrial Eye (DVS Processing)",
            # [修正] scripts/demos/ -> scripts/demos/visual/
            "cmd": f"{python_cmd} scripts/demos/visual/run_industrial_eye_demo.py"
        },
        {
            "name": "App: ECG Analysis (Temporal)",
            # [修正] scripts/ -> scripts/experiments/
            "cmd": f"{python_cmd} scripts/experiments/applications/run_ecg_analysis.py"
        },

        # --- 7. Training & Hardware ---
        {
            "name": "Train: Overfit Smoke Test",
            # [修正] scripts/trainers/ -> scripts/training/trainers/
            "cmd": f"{python_cmd} scripts/training/trainers/train_overfit_demo.py --epochs 1 --max_steps 2"
        },
        {
            "name": "Hardware: Neuromorphic Compiler",
            "cmd": f"{python_cmd} -c \"from snn_research.hardware.compiler import NeuromorphicCompiler; NeuromorphicCompiler()\""
        },
    ]

    passed = 0
    total = len(checks)

    print("-" * 60)
    for i, check in enumerate(checks):
        logger.info(f"[{i+1}/{total}] Checking {check['name']}...")
        success, _ = run_command(check["cmd"], check["name"])
        if success:
            passed += 1
    print("-" * 60)

    logger.info(f"📊 最終結果: {passed} / {total} 項目合格")

    if passed == total:
        logger.info("🎉 全システム正常稼働を確認しました。Ready for deployment.")
        sys.exit(0)
    else:
        logger.error(f"⚠️ {total - passed} 個のコンポーネントで問題が発生しています。ログを確認してください。")
        sys.exit(1)


if __name__ == "__main__":
    main()
