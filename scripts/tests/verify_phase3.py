"""
Phase 3 Verification Script
Verifies the functionality of SFormer and SEMM models on the active device (CPU/MPS/CUDA).
"""

import logging
import os
import sys

import torch

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# SFormerのインポート
try:
    from snn_research.models.transformer.sformer import SFormer
except ImportError:
    SFormer = None  # type: ignore

# SEMM (SEMMModel) のインポート
try:
    from snn_research.models.experimental.semm_model import SEMMModel
except ImportError:
    SEMMModel = None  # type: ignore

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase3Verify")


def get_device() -> torch.device:
    """利用可能な最適なデバイス（MPS > CUDA > CPU）を取得します。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def verify_phase3_models():
    """Phase 3のモデル（SFormer, SEMM）の動作検証を行います。"""
    print("=== SNN Phase 3 Completion Verification ===")
    
    device = get_device()
    print(f"Running on Device: {device}")
    
    # テスト用の語彙サイズと次元数
    vocab_size = 1000
    d_model = 64

    # --- 1. SFormer Verification ---
    if SFormer is not None:
        try:
            print(">>> Testing SFormer...")
            # 修正: vocab_sizeを引数として渡す
            model = SFormer(vocab_size=vocab_size, d_model=d_model).to(device)
            model.eval()

            # ダミー入力の生成 (Batch, Seq_Len)
            # MPSエラー回避のため、deviceへの転送を確実に行う
            dummy_input = torch.randint(0, vocab_size, (1, 32)).to(device)

            with torch.no_grad():
                # SFormer returns: logits, avg_spikes, mem
                output_tuple = model(dummy_input)
                output = output_tuple[0]  # logits

            print(f"✅ SFormer Verified (Output Shape: {output.shape})")

        except Exception as e:
            logger.error(f"❌ SFormer Verification Failed: {e}")
            if "Placeholder storage" in str(e):
                logger.error("   Hint: Input tensors must be on the same device (MPS) as the model.")
    else:
        logger.warning("⚠️ SFormer class could not be imported. Skipping.")

    # --- 2. SEMM Verification ---
    if SEMMModel is not None:
        try:
            print(">>> Testing SEMM (SEMMModel)...")
            # 修正: SEMMModelを使用し、vocab_sizeを渡す
            model = SEMMModel(vocab_size=vocab_size, d_model=d_model).to(device)
            model.eval()

            # ダミー入力の生成
            dummy_input = torch.randint(0, vocab_size, (1, 32)).to(device)

            with torch.no_grad():
                # SEMMModel returns: logits, avg_spikes, mem, aux_loss_logits
                output_tuple = model(dummy_input)
                output = output_tuple[0] # logits

            print(f"✅ SEMM Verified (Output Shape: {output.shape})")

        except Exception as e:
            logger.error(f"❌ SEMM Verification Failed: {e}")
    else:
        logger.warning("⚠️ SEMMModel class could not be imported. Skipping.")

    print("\n=== Verification Complete ===")


if __name__ == "__main__":
    verify_phase3_models()