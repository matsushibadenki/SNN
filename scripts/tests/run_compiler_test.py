# ファイルパス: scripts/runners/run_compiler_test.py
from snn_research.hardware.compiler import NeuromorphicCompiler
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter
import sys
import os
import logging

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-15s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("CompilerTest")


def main():
    logger.info("==================================================")
    logger.info("   Brain v2.0: Hardware Deployment Test          ")
    logger.info("==================================================")

    # 1. コンパイラ初期化
    compiler = NeuromorphicCompiler(
        target_hardware="Matsushiba_Neuromorphic_Chip_v1")

    # 2. モデルロード (Brain v2.0の中核モデル)
    logger.info(">>> Loading Brain v2.0 System 1 (Mamba)...")
    mamba_config = {"d_model": 128, "d_state": 32,
                    "num_layers": 4, "tokenizer": "gpt2"}

    # アダプターからモデル実体を取り出す
    adapter = AsyncBitSpikeMambaAdapter(mamba_config, device="cpu")
    model = adapter.model

    if model is None:
        logger.error("❌ Model not found. Please train the model first.")
        return

    # 3. コンパイル実行
    logger.info(">>> Starting Compilation Pipeline...")
    output_path = "workspace/deployment/Brain_v20_Core_manifest.json"
    stats = compiler.compile(model, output_path=output_path)

    # 4. レポート表示
    logger.info("\n📊 --- Deployment Report ---")
    logger.info(f"   Target Hardware: {compiler.target}")
    logger.info(f"   Total Neurons:   {stats['total_neurons']}")
    logger.info(f"   Total Synapses:  {stats['total_synapses']}")
    logger.info(f"   Est. Power:      {stats['estimated_power_mW']:.2f} mW")
    logger.info(
        f"   Required Cores:  {stats['core_mapping']['required_cores']}")
    logger.info("   Quantization:    Int8 (Dynamic Scaling)")

    logger.info("\n🚀 Ready for Flash Memory Write.")
    logger.info(">>> Deployment Simulation Finished Successfully.")


if __name__ == "__main__":
    main()
