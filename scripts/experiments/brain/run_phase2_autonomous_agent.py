# ファイルパス: scripts/experiments/brain/run_phase2_autonomous_agent.py
# 日本語タイトル: Phase 2 自律統合エージェント (Scaled & Multimodal) v1.4
# 目的: Phase 2の成果統合。記憶形成と睡眠固定化の連携を修正。
# 修正履歴:
#   v1.4: act_and_adapt内で sleep_system.store_experience を呼び出し、記憶を保存するように変更。

import sys
import os
import time
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, Union

# プロジェクトルートの設定
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ロギング設定 (force=Trueで強制適用)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Agent] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger("Phase2Agent")

# 必要なモジュールのインポート
try:
    from snn_research.core.snn_core import SNNCore
    from snn_research.adaptive.intrinsic_motivator import IntrinsicMotivator
    from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
    from snn_research.io.universal_encoder import UniversalSpikeEncoder
except ImportError as e:
    logger.error(f"❌ Import Error: {e}")
    print("Please ensure you are running this script from the project root or correct path.")
    sys.exit(1)


class Phase2AutonomousAgent:
    """
    Phase 2 目標達成のための統合自律エージェント。
    特徴:
    1. Scaled Brain: d_model=512 の大規模SFormerを使用。
    2. Multimodal: 視覚とテキストを統合処理。
    3. Autonomous: 好奇心と睡眠サイクルによる自律制御。
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🧠 Initializing Phase 2 Agent on {self.device}...")

        # 1. 脳の構築 (Scaled SFormer)
        self.brain_config = {
            "architecture_type": "sformer",  # Spiking Transformer
            "d_model": 512,                  # Scale Up Goal
            "num_layers": 4,
            "nhead": 8,
            "time_steps": 8,
            "neuron_config": {"type": "lif", "v_threshold": 1.0},
            "vocab_size": 1000
        }
        self.brain = SNNCore(config=self.brain_config,
                             vocab_size=1000).to(self.device)
        logger.info(
            f"   -> Brain Model: SFormer (d_model={self.brain_config['d_model']}) initialized.")

        # 2. 感覚器 (Universal Encoder)
        self.encoder = UniversalSpikeEncoder()

        # 3. 本能 (Curiosity & Motivation)
        self.motivator = IntrinsicMotivator(
            config={"curiosity_threshold": 0.3})

        # 4. 恒常性 (Sleep System)
        self.sleep_system = SleepConsolidator(
            target_brain_model=self.brain
        )

        # 状態管理
        self.fatigue_level = 0.0
        self.knowledge_base = []
        self.step_count = 0

    def perceive(self, sensory_input: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """多感覚統合プロセス"""
        # 視覚情報のエンコード
        if "image" in sensory_input:
            visual_spikes = self.encoder.encode(
                sensory_input["image"], modality="image"
            ).to(self.device)
        else:
            visual_spikes = torch.zeros(1, 8, 512).to(self.device)

        # テキスト/概念情報のエンコード
        if "text_id" in sensory_input:
            concept_input = torch.tensor(
                [[sensory_input["text_id"]]]).to(self.device)
        else:
            concept_input = torch.zeros(1, 1).long().to(self.device)

        return visual_spikes, concept_input

    def think(self, visual_spikes: torch.Tensor, concept_input: torch.Tensor) -> torch.Tensor:
        """思考プロセス (推論)"""
        start_time = time.time()

        # 脳による処理
        raw_output = self.brain(concept_input)
        
        output_spikes: torch.Tensor
        if isinstance(raw_output, tuple):
            output_spikes = raw_output[0]
        elif isinstance(raw_output, dict):
             if 'logits' in raw_output:
                 output_spikes = raw_output['logits']
             elif 'spikes' in raw_output:
                 output_spikes = raw_output['spikes']
             else:
                 output_spikes = list(raw_output.values())[0] # type: ignore
        else:
            output_spikes = raw_output

        latency = (time.time() - start_time) * 1000
        logger.info(f"   ⚡ Thought Latency: {latency:.2f} ms")

        return output_spikes

    def act_and_adapt(self, output: torch.Tensor):
            """行動と適応（好奇心・疲労・記憶形成）"""
            try:
                if output.numel() > 1:
                    novelty = torch.var(output.float()).item()
                else:
                    novelty = 0.0
            except Exception:
                novelty = 0.0

            # 好奇心判定 (簡易閾値)
            is_curious = novelty > 0.01

            if is_curious:
                logger.info("   🔍 Curiosity Triggered! Forming new memory...")
                self.knowledge_base.append("New Pattern Discovered")
            
                # [修正箇所]
                # output(logits/float) をそのまま保存せず、予測トークンID(long)に変換して保存する。
                # これにより、睡眠学習(SFormer)でのEmbedding入力エラーを防ぎます。
                with torch.no_grad():
                    # argmaxで最も可能性の高いIDを取得し、CPUへ転送
                    dummy_state = torch.argmax(output, dim=-1).long().cpu()
                
                    # 万が一形状がスカラーになってしまった場合の次元調整
                    if dummy_state.dim() == 0:
                        dummy_state = dummy_state.unsqueeze(0)
                    if dummy_state.dim() == 1:
                        dummy_state = dummy_state.unsqueeze(0)  # (1, SeqLen)

                    dummy_text = torch.tensor([1]).cpu() # Dummy ID
                    reward_val = 1.0 # Positive reward for curiosity
                
                    self.sleep_system.store_experience(
                        image=dummy_state, # ここで変換済みのIDを渡す
                        text=dummy_text,
                        reward=reward_val
                    )

            # 疲労の蓄積
            self.fatigue_level += 0.1
            logger.info(f"   🔋 Fatigue Level: {self.fatigue_level:.1f}/1.0")
            
    def run_life_cycle(self, max_steps: int = 15):
        """自律ライフサイクルの実行"""
        logger.info("🚀 Starting Autonomous Life Cycle")

        for step in range(max_steps):
            self.step_count += 1
            print(f"\n--- Step {self.step_count} ---")

            # 0. 睡眠チェック
            if self.fatigue_level >= 0.8:
                logger.info("💤 Fatigue limit reached. Initiating Sleep...")
                
                # 記憶がある状態で睡眠を実行
                summary = self.sleep_system.perform_sleep_cycle(
                    duration_cycles=3
                )
                
                # 睡眠結果の表示
                if summary.get("status") == "success":
                    consolidated = summary.get("consolidated_to_cortex", 0)
                    logger.info(f"   -> Sleep Successful: Consolidated {consolidated} memories to Long-term Cortex.")
                else:
                    logger.info(f"   -> Sleep Finished: {summary}")
                
                self.fatigue_level = 0.0
                continue

            # 1. 環境からの入力
            dummy_input = {
                "image": torch.randn(1, 3, 32, 32),
                "text_id": np.random.randint(0, 1000)
            }

            # 2. 知覚
            vis, txt = self.perceive(dummy_input)

            # 3. 思考
            output = self.think(vis, txt)

            # 4. 行動と適応
            self.act_and_adapt(output)

            time.sleep(0.1)

        logger.info("🏁 Life cycle simulation completed.")


if __name__ == "__main__":
    try:
        agent = Phase2AutonomousAgent()
        agent.run_life_cycle(max_steps=15)
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")
    except Exception as e:
        logger.error(f"❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()