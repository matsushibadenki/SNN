# ファイルパス: scripts/verify_phase3.py
# Title: Phase 3 統合検証スクリプト
# Description:
#   ROADMAP Phase 3 の主要成果物である SFormer, SEMM, Causal Visual Cortex の
#   動作検証を行う。

import torch
import sys
from pathlib import Path
import logging

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_research.core.snn_core import SNNCore
from snn_research.models.bio.visual_cortex import VisualCortex
from snn_research.training.losses import CombinedLoss
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase3Verify")

def verify_sformer():
    logger.info("--- 1. Verifying SFormer (T=1 Backbone) ---")
    config = {
        "architecture_type": "sformer",
        "d_model": 64,
        "num_layers": 2,
        "n_head": 2,
        "dim_feedforward": 128,
        "time_steps": 1, # Must be 1
        "neuron": {"type": "scale_and_fire", "base_threshold": 1.0}
    }
    
    model = SNNCore(config, vocab_size=100)
    dummy_input = torch.randint(0, 100, (4, 16)) # (Batch, Seq)
    
    try:
        outputs = model(dummy_input, return_spikes=True)
        logits = outputs[0]
        avg_spikes = outputs[1]
        
        logger.info("✅ SFormer Forward Pass Success.")
        logger.info(f"   Output Shape: {logits.shape}")
        logger.info(f"   Avg Spikes (SFN Activity): {avg_spikes.item():.4f}")
        
        if model.model.time_steps != 1:
            logger.error(f"❌ SFormer time_steps is {model.model.time_steps}, expected 1.")
        else:
            logger.info("✅ SFormer confirmed running at T=1.")
            
    except Exception as e:
        logger.error(f"❌ SFormer Verification Failed: {e}")

def verify_semm():
    logger.info("\n--- 2. Verifying SEMM (Spiking MoE) ---")
    config = {
        "architecture_type": "semm",
        "d_model": 64,
        "num_layers": 2,
        "num_experts": 4,
        "top_k": 2,
        "time_steps": 1,
        "neuron": {"type": "lif"}
    }
    
    model = SNNCore(config, vocab_size=100)
    dummy_input = torch.randint(0, 100, (4, 16))
    
    try:
        # SEMM returns (logits, spikes, mem, aux_logits)
        outputs = model(dummy_input, return_spikes=True)
        logits = outputs[0]
        aux_logits = outputs[3]
        
        logger.info("✅ SEMM Forward Pass Success.")
        logger.info(f"   Logits Shape: {logits.shape}")
        logger.info(f"   Aux Logits Shape (Routing Info): {aux_logits.shape}")
        
        # Loss check
        dummy_target = torch.randint(0, 100, (4, 16))
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        loss_fn = CombinedLoss(tokenizer, moe_load_balancing_weight=0.1)
        
        loss_dict = loss_fn(logits, dummy_target, outputs[1], outputs[2], model, aux_logits=aux_logits)
        moe_loss = loss_dict.get('moe_loss', 0.0)
        
        logger.info(f"✅ Load Balancing Loss Calculated: {moe_loss:.6f}")
        
    except Exception as e:
        logger.error(f"❌ SEMM Verification Failed: {e}")

def verify_causal_perception():
    logger.info("\n--- 3. Verifying Causal Perception (Visual Cortex) ---")
    # DVS入力を想定 (Batch, Time, Channels, Height, Width)
    dummy_dvs_input = torch.randn(2, 16, 2, 32, 32) 
    
    model = VisualCortex(
        input_channels=2,
        height=32,
        width=32,
        d_model=64,
        d_state=32,
        time_steps=16 # Visual Cortexは時間発展が必要
    )
    
    try:
        # forward returns (causal_state, prediction_error, reconstruction)
        states, errors, recons = model(dummy_dvs_input)
        
        logger.info("✅ Visual Cortex Forward Pass Success.")
        logger.info(f"   Causal State Shape: {states.shape} (Should be B, T, D_state)")
        logger.info(f"   Prediction Error Shape: {errors.shape}")
        
        # 因果状態の変動を確認 (時間変化しているか)
        state_variance = states.var(dim=1).mean().item()
        logger.info(f"   State Temporal Variance: {state_variance:.4f} (Should be > 0)")
        
        if state_variance > 0:
            logger.info("✅ Causal states are dynamically evolving over time.")
        else:
            logger.warning("⚠️ Causal states seem static. Check neuron dynamics.")
            
    except Exception as e:
        logger.error(f"❌ Visual Cortex Verification Failed: {e}")

if __name__ == "__main__":
    print("=== SNN Phase 3 Completion Verification ===")
    verify_sformer()
    verify_semm()
    verify_causal_perception()
    print("\n=== Verification Complete ===")