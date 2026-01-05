# ファイルパス: scripts/runners/run_post_backprop_demo.py

from snn_research.core.snn_core import SNNCore
from snn_research.core.networks.bio_pc_network import BioPCNetwork
from snn_research.learning_rules.bcm_rule import BCMLearningRule
from snn_research.adaptive.test_time_adaptation import TestTimeAdaptationWrapper
import logging
import torch  # E402 fixed
import sys
import os

# ------------------------------------------------------------------------------
# [Auto-inserted by fix_script_paths.py]
# プロジェクトルートディレクトリをsys.pathに追加して、snn_researchモジュールを解決可能にする
# このファイルは scripts/runners/ に配置されていることを想定しています (ルートから2階層下)
# ------------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------------------

# ファイルパス: run_post_backprop_demo.py
# Title: Post-Backpropagation Demo (Fixed Image Size & Logging)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s', stream=sys.stdout, force=True)
logger = logging.getLogger("Demo")


def demo_test_time_adaptation():
    logger.info("\n=== 1. TTA Demo ===")
    config = {"architecture_type": "spiking_cnn", "num_classes": 10,
              "time_steps": 16, "neuron": {"type": "lif"}}
    model = SNNCore(config, vocab_size=10)
    bcm = BCMLearningRule(0.01, 50.0, 0.1)
    tta_model = TestTimeAdaptationWrapper(
        model, bcm, adaptation_rate_multiplier=0.5)

    dummy_input = torch.randn(4, 3, 224, 224)  # Correct size for SpikingCNN
    noisy_input = dummy_input + torch.randn_like(dummy_input)

    tta_model.eval()
    with torch.no_grad():
        out = tta_model(dummy_input)
        logger.info(f"Normal Output Mean: {out[0].mean().item():.4f}")
        for i in range(3):
            out = tta_model(noisy_input)
            logger.info(
                f"Noisy Step {i+1} Output Mean: {out[0].mean().item():.4f} (Adapting...)")


def demo_bio_pc_network():
    logger.info("\n=== 2. Bio-PCNet Demo ===")
    pc_net = BioPCNetwork([128, 64, 10], 16, {
                          "type": "lif", "tau_mem": 10.0}, 0.005)
    input_data = torch.randn(8, 128)
    target = torch.randn(8, 10)

    for ep in range(3):
        pc_net.reset_state()
        _ = pc_net(input_data, targets=target)  # Target Clamping
        metrics = pc_net.run_learning_step(input_data, target)
        mag = sum(v.item() for k, v in metrics.items() if 'magnitude' in k)
        logger.info(f"Epoch {ep+1}: Update Mag = {mag:.4f}")


if __name__ == "__main__":
    demo_test_time_adaptation()
    demo_bio_pc_network()
