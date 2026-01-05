# ファイルパス: scripts/run_vlm_adaptation.py
# (Phase 4: Autonomous Adaptation - Demo)
# Title: VLM Test-Time Adaptation Demo
# Description:
#   学習済みSpikingVLMを用い、推論時のオンチップ適応（Test-Time Adaptation）を実証する。
#   不確実性が高い入力に対して、OnChipSelfCorrectorがニューロンの閾値を動的に調整する様子を観察する。

from snn_research.data.datasets import ImageTextDataset
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.adaptive.on_chip_self_corrector import OnChipSelfCorrector
from snn_research.models.transformer.spiking_vlm import SpikingVLM
import sys
import os
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer

# プロジェクトルートをPythonパスに追加
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


# ロガー設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def load_trained_model(checkpoint_path, device, vocab_size=30522, d_model=256, vision_dim=128):
    """学習済みモデルの構築と重みロード"""
    logger.info(f"📂 Loading checkpoint: {checkpoint_path}")

    vision_config = {
        "architecture_type": "spiking_cnn",
        "input_channels": 3,
        "features": vision_dim,
        "time_steps": 4,
        "layers": [64, 128, vision_dim]
    }
    language_config = {
        "architecture_type": "spiking_transformer",
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_layers": 4,
        "num_heads": 4,
        "time_steps": 4,
        "max_len": 64
    }
    projector_config = {"visual_dim": vision_dim,
                        "use_bitnet": True}  # BitNet有効化

    model = SpikingVLM(
        vocab_size=vocab_size,
        vision_config=vision_config,
        language_config=language_config,
        projector_config=projector_config
    ).to(device)

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info("✅ Model weights loaded successfully.")
    except FileNotFoundError:
        logger.warning(
            "⚠️ Checkpoint not found. Using random weights for demonstration.")

    return model


def collect_monitor_neurons(model):
    """モデル内のAdaptiveLIFNeuronを収集する"""
    neurons = []
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveLIFNeuron):
            neurons.append(module)
    logger.info(f"🔍 Found {len(neurons)} adaptive neurons to monitor.")
    return neurons


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "workspace/checkpoints/vlm/spiking_vlm_epoch_2.pt"  # 前回の学習結果
    data_path = "data/vlm_dummy/train_data.jsonl"

    # 1. モデル準備
    model = load_trained_model(checkpoint_path, device)
    model.eval()  # 推論モード (しかし適応は動く)

    # 2. オンチップ自己修正器の初期化
    monitor_neurons = collect_monitor_neurons(model)
    corrector = OnChipSelfCorrector(
        monitor_layers=monitor_neurons,
        adaptation_rate=0.05,  # デモ用に高めに設定 (変化をわかりやすくするため)
        entropy_threshold=0.5,  # 閾値を低めに設定して適応を誘発
        homeostasis_target=0.1  # 目標発火率
    )

    # 3. データローダー (1バッチずつ処理)
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception:
        # ダミートークナイザー
        class DummyTokenizer:
            pad_token_id = 0

            def __call__(self, text, **kwargs):
                ids = [hash(w) % 30522 for w in text.split()]
                return {"input_ids": torch.tensor([ids[:64]], dtype=torch.long)}
        tokenizer = DummyTokenizer()

    dataset = ImageTextDataset(data_path, tokenizer, max_seq_len=64)

    # 4. 適応推論ループ
    logger.info("🚀 Starting Test-Time Adaptation Loop...")

    entropy_history = []
    threshold_history = []  # 最初のニューロンの閾値を記録

    # 最初のニューロンの閾値パラメータへの参照
    target_neuron = monitor_neurons[0] if monitor_neurons else None

    for i in tqdm(range(min(20, len(dataset)))):  # 20ステップだけ実行
        item = dataset[i]

        # データ準備 (Batch dim追加)
        input_ids = item['input_ids'].unsqueeze(0).to(device)
        pixel_values = item['pixel_values'].unsqueeze(0).to(device)

        # --- 推論 ---
        with torch.no_grad():
            logits, avg_spikes, _ = model(
                input_ids, pixel_values, return_spikes=True)

            # --- オンチップ適応 ---
            # hidden_statesは現在未使用なので空リストを渡す
            stats = corrector(logits, hidden_states=[])

        # 記録
        entropy_history.append(stats.get("entropy", 0.0))
        if target_neuron is not None:
            # 現在の実効閾値 (Base + Adaptive)
            current_th = target_neuron.base_threshold.mean().item()
            threshold_history.append(current_th)

    # 結果表示
    logger.info(f"📊 Final Entropy: {entropy_history[-1]:.4f}")
    logger.info(f"📊 Adaptation Count: {corrector.adaptation_count.item()}")

    if target_neuron:
        logger.info(
            f"📈 Threshold Change: {threshold_history[0]:.4f} -> {threshold_history[-1]:.4f}")
        if threshold_history[-1] != threshold_history[0]:
            logger.info(
                "✅ SUCCESS: Neuron thresholds adapted dynamically during inference!")
        else:
            logger.info(
                "ℹ️ No significant threshold change (Entropy might be low).")


if __name__ == "__main__":
    main()
