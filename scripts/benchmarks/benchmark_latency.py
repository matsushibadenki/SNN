# ファイルパス: scripts/benchmarks/benchmark_latency.py
# 日本語タイトル: Benchmark Latency Tool
# 目的: モデルの推論レイテンシを測定し、ログに出力する。

import time
import torch
import logging
import sys
import os
from omegaconf import OmegaConf
from snn_research.core.snn_core import SNNCore

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def benchmark(config_path: str):
    print(f"🚀 Benchmarking config: {config_path}")
    logger.info(f"🚀 Benchmarking config: {config_path}")

    if not os.path.exists(config_path):
        error_msg = f"❌ Config file not found: {config_path}"
        logger.error(error_msg)
        print(error_msg)
        # フォールバック: デフォルト設定が存在しない場合は終了せず警告
        return None

    try:
        # Load config
        conf = OmegaConf.load(config_path)

        # Initialize model
        vocab_size = 1000
        # config構造の堅牢性チェック
        model_conf = conf.model if hasattr(conf, 'model') else conf

        logger.info("Initializing SNNCore...")
        model = SNNCore(config=model_conf, vocab_size=vocab_size,
                        backend="spikingjelly")
        model.eval()

        # Dummy input
        batch_size = 1
        seq_len = getattr(model_conf, 'time_steps', 16)  # デフォルト値
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Warmup
        logger.info("Warmup...")
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_ids)

        # Measure latency
        logger.info("Measuring latency...")
        latencies = []
        with torch.no_grad():
            for _ in range(20):
                start_time = time.time()
                _ = model(input_ids)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # ms

        avg_latency = sum(latencies) / len(latencies)
        result_msg = f"⚡️ Average Inference Latency: {avg_latency:.2f} ms"
        logger.info(result_msg)
        print(result_msg)

        with open("latency_result.txt", "w") as f:
            f.write(f"Average Inference Latency: {avg_latency:.2f} ms\n")

        return avg_latency

    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    config_path = sys.argv[1] if len(
        sys.argv) > 1 else "configs/models/large_scale.yaml"

    # プロジェクトルートからの相対パス補正
    if not os.path.exists(config_path):
        # 試しにプロジェクトルート直下を探す
        potential_path = os.path.join(os.getcwd(), config_path)
        if os.path.exists(potential_path):
            config_path = potential_path

    benchmark(config_path)
