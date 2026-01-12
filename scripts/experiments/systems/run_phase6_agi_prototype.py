# ファイルパス: scripts/experiments/systems/run_phase6_agi_prototype.py
# 日本語タイトル: Phase 6 AGIプロトタイプ "Genesis"
# 目的: 全フェーズの成果(視覚・ハイブリッド脳・睡眠・社会性)を統合した自律進化型エージェントの実装。

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
    format='%(asctime)s - [Genesis] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
# 外部ライブラリのログ抑制
logging.getLogger("spikingjelly").setLevel(logging.ERROR)

# 必要なモジュールのインポート
try:
    from snn_research.core.snn_core import SNNCore
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
    from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
    from snn_research.adaptive.intrinsic_motivator import IntrinsicMotivator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


class GenesisBrain(nn.Module):
    """
    AGIプロトタイプ用ハイブリッド脳。
    """
    def __init__(self, device: str, vocab_size: int = 128):
        super().__init__()
        self.device = device
        
        # System 1: SFormer (高速思考)
        self.system1 = SNNCore(config={
            "architecture_type": "sformer",
            "d_model": 64,
            "num_layers": 2,
            "nhead": 2,
            "time_steps": 2,
            "neuron_config": {"type": "lif", "v_threshold": 1.0}
        }, vocab_size=vocab_size).to(device)
        
        # System 2: BitSpikeMamba (深層思考)
        self.system2 = BitSpikeMamba(
            vocab_size=vocab_size,
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
            num_layers=2,
            time_steps=4,
            neuron_config={"type": "lif", "base_threshold": 1.0}
        ).to(device)
        
        self.classifier = nn.Linear(vocab_size, 10).to(device)

    def forward(self, x: torch.Tensor, use_system2: bool = False) -> Dict[str, Any]:
        if not x.is_contiguous(): x = x.contiguous()
        
        # System 1
        out1 = self.system1(x)
        if isinstance(out1, tuple): out1 = out1[0]
        features = out1.mean(dim=1)
        
        system = "System 1"
        
        # System 2 Override
        if use_system2:
            system = "System 2"
            out2 = self.system2(x)
            if isinstance(out2, tuple): out2 = out2[0]
            # 特徴統合 (簡易的に平均)
            features = (features + out2.mean(dim=1)) / 2.0
            
        logits = self.classifier(features)
        return {"logits": logits, "system": system, "features": features}


class GenesisAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logging.info(f"🚀 Initializing AGI Prototype 'Genesis' on {self.device}...")
        
        self.brain = GenesisBrain(self.device).to(self.device)
        self.motivator = IntrinsicMotivator()
        self.sleep_system = SleepConsolidator(target_brain_model=self.brain.system2)
        
        self.age = 0
        self.fatigue = 0.0
        self.knowledge = 0
        self.state = "Awake"

    def live(self, steps: int = 50):
        logging.info("🌍 Genesis is now alive. Exploring the digital void...")
        
        try:
            for step in range(1, steps + 1):
                self.age += 1
                
                # 1. 環境入力 (シミュレーション: Webからの情報など)
                # ランダムなトークン列 (Batch=1, Seq=8)
                input_data = torch.randint(0, 128, (1, 8)).to(self.device)
                
                # 2. 思考プロセス
                start_time = time.time()
                
                # 複雑度判定 (ランダム)
                is_complex = (np.random.random() < 0.2)
                
                result = self.brain(input_data, use_system2=is_complex)
                latency = (time.time() - start_time) * 1000
                
                # 3. 内部状態更新
                prediction = torch.argmax(result["logits"], dim=-1).item()
                novelty = torch.var(result["features"].float()).item()
                
                if novelty > 0.1 or is_complex:
                    self.knowledge += 1
                    self.fatigue += 0.15
                    log_msg = f"💡 Insight! (Nov:{novelty:.2f})"
                    
                    # 記憶の保存
                    mem_tokens = input_data.cpu()
                    self.sleep_system.store_experience(mem_tokens, torch.tensor([prediction]), 1.0)
                else:
                    self.fatigue += 0.05
                    log_msg = "Thinking..."

                # 4. ログ出力
                sys_name = result["system"]
                print(f"Age {self.age:03} | {sys_name:<8} | {latency:6.2f}ms | Fat:{self.fatigue:4.2f} | Know:{self.knowledge:3} | {log_msg}")
                
                # 5. 睡眠サイクル
                if self.fatigue >= 1.0:
                    self.sleep()
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            logging.info("🛑 Genesis saved state and shut down.")
        except Exception as e:
            logging.error(f"❌ Genesis crashed: {e}")
            import traceback
            traceback.print_exc()

    def sleep(self):
        logging.info("💤 Fatigue limit reached. Entering REM sleep...")
        self.state = "Sleeping"
        
        summary = self.sleep_system.perform_sleep_cycle(duration_cycles=3)
        
        logging.info(f"   -> Dream Replay: {summary.get('avg_replay_loss', 0):.4f} loss")
        logging.info(f"   -> Consolidation: {summary.get('consolidated_to_cortex', 0)} memories fixed.")
        
        self.fatigue = 0.0
        self.state = "Awake"
        logging.info("🌅 Genesis woke up evolved.")


if __name__ == "__main__":
    ai = GenesisAgent()
    ai.live(steps=60)