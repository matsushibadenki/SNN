# ファイルパス: scripts/demos/systems/run_multimodal_vlm_demo.py
# 日本語タイトル: Multimodal VLM Training & Inference Demo (Log Fix)
# 目的・内容:
#   SpikingVLM と MultimodalTrainer を統合し、ダミーデータを用いて
#   学習ループが正常に回ること、および推論（キャプション生成）が可能であることを実証する。
#   [Fix] ログが表示されない問題を修正 (force=True)。

from snn_research.training.trainers.multimodal_trainer import MultimodalVLMTrainer
from snn_research.core.architecture_registry import ArchitectureRegistry
import os
import sys
import torch
import logging
from torch.utils.data import DataLoader, Dataset

# プロジェクトルートへのパス追加
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

# ログ設定: force=True で既存の設定を上書きし、確実に表示させる
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


class DummyMultimodalDataset(Dataset):
    """
    テスト用のダミーデータセット。
    ランダムな画像とトークンIDを返す。
    """

    def __init__(self, size: int = 100, seq_len: int = 16, img_size: int = 32):
        self.size = size
        self.seq_len = seq_len
        self.img_size = img_size
        self.vocab_size = 1000  # Demo用

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random Image: [3, 32, 32]
        image = torch.randn(3, self.img_size, self.img_size)
        # Random Text: [SeqLen]
        text = torch.randint(1, self.vocab_size, (self.seq_len,))
        # Labels (Same as text for simple autoregressive test)
        labels = text.clone()

        return {
            "images": image,
            "input_ids": text,
            "labels": labels
        }


def run_demo():
    logger.info("🎬 Starting Multimodal VLM Demo...")

    # 1. Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 1000
    img_size = 32
    time_steps = 4  # SNN Time Steps

    config = {
        "vision_config": {
            "type": "cnn",
            "hidden_dim": 128,
            "img_size": img_size,
            "time_steps": time_steps,
            "neuron": {"type": "lif"}
        },
        "text_config": {
            "d_model": 128,
            "vocab_size": vocab_size,
            "num_layers": 2,
            "time_steps": time_steps
        },
        "projection_dim": 128,
        "use_bitnet": False  # Demoでは軽量化のためFalseも可
    }

    # 2. Build Model using Registry (Corrected Interface)

    full_config = {
        "vision_config": config["vision_config"],
        "language_config": config["text_config"],
        "projector_config": {"projection_dim": config["projection_dim"]},
        # ビルダが期待する形式に合わせる
        "sensory_inputs": {"vision": config["vision_config"]},
        "use_bitnet": config["use_bitnet"]
    }

    logger.info("🏗️ Building SpikingVLM model via Registry...")
    try:
        model = ArchitectureRegistry.build(
            "spiking_vlm", full_config, vocab_size)
    except Exception as e:
        logger.error(f"Failed to build model via Registry: {e}")
        logger.info(
            "⚠️ Falling back to direct instantiation for demo purposes.")
        # レジストリ経由が失敗した場合の直接インスタンス化（デバッグ用）
        from snn_research.models.transformer.spiking_vlm import SpikingVLM
        model = SpikingVLM(
            vocab_size=vocab_size,
            vision_config=config["vision_config"],
            text_config=config["text_config"],
            projection_dim=config["projection_dim"],
            use_bitnet=config["use_bitnet"]
        )

    model = model.to(device)
    logger.info(f"✅ Model built successfully. Device: {device}")

    # 3. Prepare Data
    dataset = DummyMultimodalDataset(size=50, seq_len=16, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 4. Trainer Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = MultimodalVLMTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        config={
            "lambda_align": 0.5,
            "lambda_gen": 1.0
        }
    )

    # 5. Training Loop
    epochs = 2
    logger.info(f"🔄 Starting training loop for {epochs} epochs...")

    for epoch in range(epochs):
        metrics = trainer.train_epoch(dataloader, epoch)
        logger.info(f"   Epoch {epoch} Metrics: {metrics}")

    # 6. Inference / Generation Test
    logger.info("🧪 Testing Caption Generation...")
    sample_img = torch.randn(1, 3, img_size, img_size).to(device)

    try:
        generated_ids = model.generate_caption(sample_img, max_len=10)
        logger.info(f"🖼️ Input Image Shape: {sample_img.shape}")
        logger.info(f"📝 Generated Token IDs: {generated_ids.cpu().tolist()}")
        logger.info("✅ Generation executed successfully.")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("🎉 Demo completed!")


if __name__ == "__main__":
    run_demo()
