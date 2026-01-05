# ファイルパス: scripts/runners/run_brain_v21_embodiment.py
# Title: Brain v21 Embodiment Runner (Debug & Stable)
# Description:
#   ROADMAP Phase 3 Step 4: 統合と実世界テスト。
#   修正: "No Reaction" 問題への対応。
#   - Printデバッグと強制フラッシュを追加。
#   - MPSでの不安定な最適化（Channels Last, Inference Mode）を無効化。
#   - タイムステップ T=1, 軽量モデル設定は維持して高速化を図る。

import sys
import os
import time
import logging
import torch
import torch.nn as nn
from typing import Dict, Any

# --- Immediate Debug Print ---
print(f"[DEBUG] Script started. Python: {sys.version}")
sys.stdout.flush()

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

# ロギング設定 (標準出力へ強制)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Brain_v21_Embodiment")

try:
    # Phase 3 新規実装コンポーネント
    from snn_research.core.neurons.da_lif_node import DualAdaptiveLIFNode
    from snn_research.io.spike_encoder import HybridTemporal8BitEncoder
    from snn_research.models.transformer.spikformer import Spikformer, TransformerToMambaAdapter

    # 既存コンポーネント
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
    from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
    print("[DEBUG] Imports successful.")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# --- Parameters ---
TIME_STEPS = 1          # Single-Shot for lowest latency
IMG_SIZE = 128          # 128x128
PATCH_SIZE = 16
EMBED_DIM_VIS = 128     # Lightweight Vision
EMBED_DIM_PFC = 256     # Lightweight PFC

def get_optimal_device():
    """環境に合わせて最適なデバイスを選択する"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" 
    else:
        return "cpu"

class ReflexModule(nn.Module):
    """脊髄反射モジュール"""
    def __init__(self, input_dim: int, num_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=2), 
            nn.Linear(input_dim, 64), 
            DualAdaptiveLIFNode(tau_m_init=1.5, detach_reset=True),
            nn.Linear(64, num_actions),
            DualAdaptiveLIFNode(tau_m_init=1.5, detach_reset=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BrainV21(nn.Module):
    """
    Phase 3: The Spatio-Temporal Hybrid Brain (Slim Version)
    """
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        logger.info(f"🧠 Initializing Brain components on {device.upper()}...")

        # 1. Encoding
        self.encoder = HybridTemporal8BitEncoder(duration=TIME_STEPS)
        
        # 2. Visual Cortex (Spikformer) - Slim
        self.visual_cortex = Spikformer(
            img_size_h=IMG_SIZE, img_size_w=IMG_SIZE,
            patch_size=PATCH_SIZE, 
            embed_dim=EMBED_DIM_VIS,
            num_heads=4, 
            num_layers=2, T=TIME_STEPS
        ).to(device)
        
        num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
        
        # 3. Adapter
        self.adapter = TransformerToMambaAdapter(
            vis_dim=EMBED_DIM_VIS, 
            model_dim=EMBED_DIM_PFC, 
            seq_len=num_patches
        ).to(device)

        # 4. PFC (Mamba) - Slim
        self.pfc = BitSpikeMamba(
            vocab_size=1000, 
            d_model=EMBED_DIM_PFC, 
            d_state=32,
            d_conv=4,
            expand=2,
            num_layers=2,
            time_steps=TIME_STEPS,
            neuron_config={
                "type": "lif",
                "tau_mem": 2.0,
                "base_threshold": 1.0,
                "adaptation_strength": 0.1
            }
        ).to(device)
        
        # 5. Reflex
        self.reflex = ReflexModule(input_dim=num_patches*EMBED_DIM_VIS, num_actions=10).to(device)
        
        # 6. Astrocyte
        self.astrocyte = AstrocyteNetwork()

    def forward(self, visual_input: torch.Tensor) -> Dict[str, Any]:
        # Step 0: Energy Check
        if not self.astrocyte.request_resource("cortex", 5.0):
            return {"action": "REST", "reason": "Low Energy"}

        # Step 1: Encoding
        spikes = self.encoder.forward(visual_input, duration=TIME_STEPS)
        
        # Guard: Ensure dtype match (Crucial for MPS)
        target_dtype = self.visual_cortex.patch_embed.weight.dtype
        if spikes.dtype != target_dtype:
            spikes = spikes.to(dtype=target_dtype)
        
        # Step 2: Visual Perception
        visual_features = self.visual_cortex(spikes) # (B, T, N, D)
        
        # Step 3: Reflex (Fast Path)
        reflex_out = self.reflex(visual_features)
        reflex_action = reflex_out.mean(dim=1).argmax(dim=-1)
        
        # Step 4: Reasoning (Slow Path)
        context_vector = self.adapter(visual_features)
        pfc_out, _, _ = self.pfc(context_vector)
        thought_action = pfc_out[:, -1, :10].argmax(dim=-1)

        return {
            "reflex_action": reflex_action.item(),
            "thought_action": thought_action.item(),
            "energy": self.astrocyte.get_energy_level()
        }

class MockEnvironment:
    def __init__(self, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
    
    def get_observation(self) -> torch.Tensor:
        # 128x128 resolution
        return torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=self.device, dtype=self.dtype)
    
    def step(self, action: int):
        pass

def run_embodiment_test():
    device = get_optimal_device()
    print(f"[DEBUG] Selected Device: {device}")
    
    # MPS (Apple Silicon) Stable Config: Float32
    use_half = True if device == "cuda" else False
    dtype = torch.float16 if use_half else torch.float32
    precision_mode = "FP16" if use_half else "FP32"
    
    logger.info(f">>> Starting Embodiment Test on {device.upper()} [{precision_mode}]")
    sys.stdout.flush()
    
    # 1. Model Init
    try:
        brain = BrainV21(device=device)
        if use_half:
            brain = brain.half()
        
        # [DEBUG] MPSでの channels_last は不安定な場合があるため削除
        # brain = brain.to(memory_format=torch.channels_last) 
        
        brain.eval()
        logger.info("Model initialized successfully.")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return

    # 2. Env Init
    env = MockEnvironment(device=device, dtype=dtype)
    
    # 3. Warmup
    logger.info("🔥 Warming up...")
    sys.stdout.flush()
    warmup_steps = 10
    
    # [DEBUG] inference_mode -> no_grad に変更 (MPS安定性のため)
    with torch.no_grad():
        for i in range(warmup_steps):
            try:
                obs = env.get_observation()
                brain(obs)
                if device in ["cuda", "mps"]:
                    getattr(torch, device).synchronize()
            except Exception as e:
                logger.error(f"Warmup failed at step {i}: {e}")
                return

    # 4. Benchmark Loop
    num_steps = 100
    total_latency = 0.0
    
    logger.info(f"🚀 Running {num_steps} steps...")
    sys.stdout.flush()
    
    try:
        with torch.no_grad():
            for i in range(num_steps):
                if device in ["cuda", "mps"]:
                    getattr(torch, device).synchronize()
                    
                start_time = time.perf_counter()
                
                # --- Core Loop ---
                obs = env.get_observation()
                result = brain(obs)
                env.step(result["reflex_action"])
                # -----------------
                
                if device in ["cuda", "mps"]:
                    getattr(torch, device).synchronize()

                latency = (time.perf_counter() - start_time) * 1000
                total_latency += latency
                
                if i % 20 == 0:
                    logger.info(f"Step {i:03}: {latency:.2f} ms")
                    sys.stdout.flush()
                    
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.error(f"Runtime Error: {e}")
        import traceback
        traceback.print_exc()
        return

    avg_latency = total_latency / num_steps
    logger.info(f"\n>>> Average Latency: {avg_latency:.2f} ms")
    
    if avg_latency < 50.0:
        logger.info("✅ SUCCESS: Real-Time Performance Achieved! (<50ms)")
        if avg_latency < 30.0:
            logger.info("🚀 EXCELLENT: Ultra-Low Latency (<30ms) Achieved.")
    else:
        logger.warning(f"⚠️  Latency is still high. Current: {avg_latency:.2f}ms")

if __name__ == "__main__":
    torch.manual_seed(42)
    run_embodiment_test()