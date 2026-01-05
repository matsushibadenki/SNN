# ファイルパス: scripts/runners/run_brain_v20_prototype.py
# 日本語タイトル: Brain v2.6 Integration Runner - Container Managed
# 目的・内容:
#   AsyncBrainKernel v2.6 の統合テスト（最終形）。
#   修正: TrainingContainerを使用してPlannerSNNを初期化し、学習時との構成を完全統一。
#         これにより、モデル構造の不一致によるエラーを防ぎ、学習済み重みを正しくロードする。

from app.containers import TrainingContainer
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain  # E402 fixed
import sys
import os
import asyncio
import logging
import torch

# プロジェクトルートの設定
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    force=True
)
logger = logging.getLogger("BrainV2.6-Container")

# --- Mocks for other components ---


class SimulatedPerceptionModule:
    """視覚・言語兼用の知覚モジュールモック"""

    async def process(self, input_signal):
        logger.info(f"👁️/🗣️ Perception processing: {input_signal}")
        await asyncio.sleep(0.2)

        metadata = {}
        input_str = str(input_signal).lower()

        # 'plan'などのキーワードがあればPlannerへルーティング
        if "plan" in input_str or "organize" in input_str:
            metadata["needs_planning"] = True
            logger.info("   -> Detected intent: PLANNING")
        elif "unknown" in input_str:
            metadata["trigger_system2"] = True

        return {
            "payload": f"{input_signal}",  # ペイロードはそのまま渡す
            "metadata": metadata
        }


class SimulatedReasoningEngine:
    async def process(self, data):
        logger.info(f"🤔 System 2 Reasoning on: {data}")
        await asyncio.sleep(0.5)
        if "unknown" in str(data).lower():
            return {"topic": "Latest Neuromorphic Chip Architecture 2025"}
        return "RESULT: Logical conclusion derived."


class SimulatedWebCrawler:
    def crawl(self, start_url, max_pages):
        logger.info(f"🕷️ Crawling URL: {start_url}")
        return "/tmp/crawled_data_dummy.txt"


class SimulatedDistillationManager:
    async def run_on_demand_pipeline(self, task_description, unlabeled_data_path, force_retrain):
        logger.info(f"⚗️ Distilling knowledge for: {task_description}")
        await asyncio.sleep(0.5)
        return True


class SimulatedMotorCortex:
    async def process(self, command):
        if isinstance(command, dict):
            # Plannerからの詳細な出力を表示
            action = command.get('primary_action', 'Unknown')
            action_id = command.get('action_id', -1)
            logger.info(
                f"🤖 ACTUATOR: Executing Plan -> {action} (ID: {action_id})")
            logger.info(f"    Full Plan: {command.get('suggested_actions')}")
        else:
            logger.info(f"🤖 ACTUATOR: Executing command '{command}'")
        return True

# --- Main Routine ---


async def main():
    logger.info("==================================================")
    logger.info("   Matsushiba Denki SNN - Brain v2.6 Integration  ")
    logger.info("   (Container Managed Planner Initialization)     ")
    logger.info("==================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using Device: {device}")

    try:
        # 1. コンポーネントの初期化
        astrocyte = AstrocyteNetwork()

        # ★ コンテナを利用したモデル構築 ★
        # 学習時と同じ設定ファイルを読み込むことで、構成を完全に一致させる
        container = TrainingContainer()
        container.config.from_yaml("configs/templates/base_config.yaml")
        container.config.from_yaml("configs/models/small.yaml")

        # トークナイザーの取得と設定
        tokenizer = container.tokenizer()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # PlannerSNNモデルをコンテナから取得 (これで d_model, num_skills 等が自動的に合う)
        logger.info("🏗️ Building PlannerSNN from container config...")
        planner_model = container.planner_snn()
        planner_model.to(device)

        # チェックポイントのロード
        # 注意: 学習時の構成と完全に一致していれば strict=True でも通るはずだが、
        # SNN特有の内部状態(mem, spike_count等)が含まれる場合は strict=False が無難
        checkpoint_path = "workspace/runs/checkpoints/planner_epoch_15.pth"

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            try:
                planner_model.load_state_dict(
                    checkpoint['model_state_dict'], strict=False)
                logger.info(
                    f"✅ Loaded trained planner weights from {checkpoint_path}")
            except Exception as e:
                logger.error(f"⚠️ Failed to load weights: {e}")
        else:
            logger.warning(
                f"⚠️ Checkpoint not found at {checkpoint_path}. Using random weights.")

        # Real HierarchicalPlanner
        # アクションスペースは学習データ (train_planner.py) の定義に合わせる
        # 0:QA, 1:Emotion, 2:Plan
        real_planner = HierarchicalPlanner(
            planner_model=planner_model,
            tokenizer=tokenizer,
            action_space={0: "Answer Question",
                          1: "Express Emotion", 2: "Create Plan"},
            device=device
        )

        # Mamba (System 1)
        # こちらも将来的にはコンテナから取得するのが望ましい
        mamba_config = {
            "d_model": 128, "d_state": 32, "num_layers": 4, "tokenizer": "gpt2"
        }
        thinking_engine = AsyncBitSpikeMambaAdapter(
            mamba_config, device=device)

        perception = SimulatedPerceptionModule()

        # 2. Brain Kernelの構築
        logger.info(">>> Building Brain Kernel...")
        brain = AsyncArtificialBrain(
            modules={
                "visual_cortex": perception,
                "language_area": perception,
                "system1": thinking_engine,
                "reasoning_engine": SimulatedReasoningEngine(),
                "planner": real_planner,
                "actuator": SimulatedMotorCortex()
            },
            astrocyte=astrocyte,
            web_crawler=SimulatedWebCrawler(),
            distillation_manager=SimulatedDistillationManager(),
            max_workers=4
        )

        # 3. 起動
        await brain.start()
        brain.astrocyte.replenish_energy(100.0)

        # --- Scenario: Planning Task ---
        logger.info(
            "\n--- Scenario: Planning Task (Testing Real PlannerSNN) ---")
        input_text = "Please make a plan to organize the desk."
        logger.info(f"USER INPUT: {input_text}")
        await brain.receive_input(input_text)

        # 処理完了まで待機
        await asyncio.sleep(5.0)

        # --- Final Health Check ---
        await brain.stop()
        logger.info(">>> Integration Test Finished.")

    except Exception as e:
        logger.error(f"❌ Runtime Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
