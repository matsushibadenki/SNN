# ファイルパス: scripts/experiments/brain/run_phase3_hybrid_agent.py
# 日本語タイトル: Phase 3 ハイブリッド・エージェント (System 1+2 Integration) v1.2
# 目的: SFormer(直感)とBitSpikeMamba(熟考)を動的に切り替える省エネ・高性能AIの実装。
# 修正履歴:
#   v1.1: BitSpikeMambaの初期化引数不足修正。
#   v1.2: gating_networkの入力次元不一致(256 vs 1000)を修正。

import sys
import os
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional

# プロジェクトルートの設定
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [HybridAgent] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger("HybridAgent")

# 必要なモジュールのインポート
try:
    from snn_research.core.snn_core import SNNCore
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
    from snn_research.adaptive.intrinsic_motivator import IntrinsicMotivator
    from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
    from snn_research.io.universal_encoder import UniversalSpikeEncoder
except ImportError as e:
    logger.error(f"❌ Import Error: {e}")
    sys.exit(1)


class HybridBrain(nn.Module):
    """
    System 1 (Fast/SNN) と System 2 (Slow/Mamba) を統合した脳モデル。
    """
    def __init__(self, device: str, vocab_size: int = 1000):
        super().__init__()
        self.device = device
        
        # System 1: SFormer (高速・反射・低消費電力)
        logger.info("   🧠 Initializing System 1: SFormer (Fast Intuition)...")
        sformer_config = {
            "architecture_type": "sformer",
            "d_model": 256,
            "num_layers": 2,
            "nhead": 4,
            "time_steps": 4,
            "neuron_config": {"type": "lif", "v_threshold": 1.0}
        }
        self.system1 = SNNCore(config=sformer_config, vocab_size=vocab_size).to(device)
        
        # System 2: BitSpikeMamba (低ビットLLM・深い推論)
        logger.info("   🧠 Initializing System 2: BitSpikeMamba (Deep Reasoning)...")
        self.system2 = BitSpikeMamba(
            vocab_size=vocab_size,
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
            num_layers=4,
            time_steps=8,
            neuron_config={"type": "lif", "base_threshold": 1.0}
        ).to(device)
        
        # ゲート機構（どちらのシステムを使うか判断する軽量ネットワーク）
        # [修正] 入力次元を d_model(256) ではなく vocab_size(1000) に合わせる
        # なぜなら System 1 の出力(Logits)を見て判断するため。
        self.gating_network = nn.Sequential(
            nn.Linear(vocab_size, 64), 
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x: torch.Tensor, force_system2: bool = False) -> Dict[str, Any]:
        """
        入力に応じてシステムを動的に切り替えるForwardパス
        """
        # MPS対策
        if not x.is_contiguous():
            x = x.contiguous()

        # 1. まず軽量なSystem 1で特徴抽出と初期応答を生成
        # SFormerの出力を取得 (Logits)
        sys1_out = self.system1(x)
        
        if isinstance(sys1_out, tuple): sys1_out = sys1_out[0]
        
        # 2. System 2が必要か判断 (Gating)
        # ゲート判断用の特徴量 (Batch, Dim) -> (Batch, 1)
        # SFormerの出力次元が(Batch, Seq, Dim)の場合、平均を取る
        if sys1_out.dim() == 3:
            feat = sys1_out.mean(dim=1)
        else:
            feat = sys1_out
            
        gate_score = self.gating_network(feat).mean().item()
        
        used_system = "System 1"
        final_output = sys1_out
        
        # 閾値を超える、または強制フラグがあればSystem 2を起動
        if gate_score > 0.7 or force_system2:
            used_system = "System 2 (Activated)"
            # System 2 (Mamba) 実行
            sys2_out = self.system2(x)
            if isinstance(sys2_out, tuple): sys2_out = sys2_out[0]
            
            # System 1と2の統合（ここではSystem 2の結果を優先・上書き）
            final_output = sys2_out
            
        return {
            "output": final_output,
            "system": used_system,
            "gate_score": gate_score
        }


class Phase3HybridAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"🚀 Initializing Phase 3 Hybrid Agent on {self.device}...")
        
        # ハイブリッド脳の構築
        self.brain = HybridBrain(self.device).to(self.device)
        
        self.encoder = UniversalSpikeEncoder()
        self.motivator = IntrinsicMotivator(config={"curiosity_threshold": 0.3})
        
        # 睡眠システムは System 2 (長期記憶担当) を対象に最適化
        self.sleep_system = SleepConsolidator(
            target_brain_model=self.brain.system2
        )
        
        self.fatigue = 0.0
        self.steps = 0

    def perceive(self, text_id: int) -> torch.Tensor:
        # 入力を整形
        x = torch.tensor([[text_id]]).long().to(self.device)
        return x

    def run_step(self):
        self.steps += 1
        print(f"\n--- Step {self.steps} ---")
        
        # 1. 入力 (ランダムな概念ID)
        input_concept = np.random.randint(0, 1000)
        x = self.perceive(input_concept)
        
        # 2. 思考 (Hybrid Forward)
        start_time = time.time()
        
        # 時々、難解な入力(System 2が必要)が来ると仮定
        is_complex_input = (self.steps % 5 == 0) 
        
        result = self.brain(x, force_system2=is_complex_input)
        
        latency = (time.time() - start_time) * 1000
        output = result["output"]
        system_used = result["system"]
        
        logger.info(f"   🧠 Thought via {system_used}")
        logger.info(f"   ⚡ Latency: {latency:.2f} ms")
        
        # 3. 好奇心と適応
        with torch.no_grad():
            # 出力の分散を好奇心の指標とする
            novelty = torch.var(output.float()).item() if output.numel() > 1 else 0.0
            
        if novelty > 0.05 or "System 2" in system_used:
            logger.info("   🔍 Interesting concept found. Consolidating memory...")
            # System 2が動いた重要な経験を記憶する
            
            # [Fix] Memory storage with contiguous tensors for MPS
            mem_state = torch.argmax(output, dim=-1).long().cpu()
            if mem_state.dim() == 0: mem_state = mem_state.unsqueeze(0)
            if mem_state.dim() == 1: mem_state = mem_state.unsqueeze(0)
            
            self.sleep_system.store_experience(
                image=mem_state,
                text=torch.tensor([input_concept]).cpu(),
                reward=1.0
            )
            self.fatigue += 0.2 # System 2は疲れる
        else:
            self.fatigue += 0.05

        logger.info(f"   🔋 Fatigue: {self.fatigue:.2f}/1.0")

        # 4. 睡眠チェック
        if self.fatigue >= 1.0:
            logger.info("💤 Brain exhausted. Entering Deep Sleep...")
            summary = self.sleep_system.perform_sleep_cycle(duration_cycles=2)
            logger.info(f"   -> Sleep Summary: {summary}")
            self.fatigue = 0.0

    def live(self, steps=20):
        try:
            for _ in range(steps):
                self.run_step()
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user.")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    agent = Phase3HybridAgent()
    agent.live(steps=20)