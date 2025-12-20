# ファイルパス: scripts/runners/run_sleep_cycle_demo.py
# 日本語タイトル: SNN Sleep & Consolidation Demo (Type Safe v2)
# 目的・内容:
#   ROADMAP v16.3 "Sleep & Memory Consolidation" の動作検証。
#   修正: simulate_daytime_experiences 内での mypy 型エラー(arg-type)を修正。
#   [Fix] KeyError: 'energy_percent' 対策 (辞書アクセスを安全に)。

import os
import sys
import torch
import logging
import time
import random
from typing import Dict, Any, cast, List

# プロジェクトルートの設定
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# ログ設定
from app.utils import setup_logging
logger = setup_logging(log_dir="logs", log_name="sleep_cycle_demo.log")

# --- Import SNN Modules ---
from snn_research.core.snn_core import SNNCore
from snn_research.models.transformer.sformer import SFormer
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator

# Cognitive Components
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.hybrid_perception_cortex import HybridPerceptionCortex
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia

# Mock
class MockComponent:
    def __init__(self, name): self.name = name
    def __call__(self, *args, **kwargs): return None
    def select_action(self, *args): return {'action': 'sleep', 'params': {}}

def build_brain_with_sleep(device='cpu') -> ArtificialBrain:
    logger.info("🧠 Initializing Artificial Brain with Sleep Capabilities...")

    # 1. Core Models
    vocab_size = 50300
    model_config = {
        'd_model': 128, 'num_layers': 2, 'nhead': 4,
        'vocab_size': vocab_size, 'architecture_type': 'sformer'
    }
    thinking_engine = SNNCore(model_config, vocab_size=vocab_size).to(device)
    
    # 2. Memory & Sleep System
    # 簡易RAG (None) と SNNモデルを結合
    memory_system = Memory(rag_system=None, memory_path="runs/demo_sleep_memory.jsonl") # type: ignore
    
    # SleepConsolidatorの初期化
    sleep_manager = SleepConsolidator(
        memory_system=memory_system,
        target_brain_model=thinking_engine,
        device=device
    )
    
    # 3. Homeostasis (Start with low energy to force sleep need)
    astrocyte = AstrocyteNetwork(max_energy=1000.0, fatigue_threshold=80.0)
    astrocyte.current_energy = 200.0 # Low energy
    astrocyte.fatigue_toxin = 90.0   # High fatigue

    # 4. Other Components
    workspace = GlobalWorkspace()
    perception = HybridPerceptionCortex(workspace=workspace, num_neurons=128)
    basal = BasalGanglia(workspace=workspace)
    motivation = IntrinsicMotivationSystem()

    # 5. Brain Assembly
    brain = ArtificialBrain(
        global_workspace=workspace,
        motivation_system=motivation,
        sensory_receptor=SensoryReceptor(),
        spike_encoder=SpikeEncoder(),
        actuator=Actuator(actuator_name="sleeper_agent"),
        thinking_engine=thinking_engine,
        perception_cortex=perception,
        visual_cortex=MockComponent("Visual"), # type: ignore
        prefrontal_cortex=MockComponent("PFC"), # type: ignore
        hippocampus=MockComponent("Hippocampus"), # type: ignore
        cortex=MockComponent("Cortex"), # type: ignore
        amygdala=MockComponent("Amygdala"), # type: ignore
        basal_ganglia=basal,
        cerebellum=MockComponent("Cerebellum"), # type: ignore
        motor_cortex=MockComponent("Motor"), # type: ignore
        causal_inference_engine=MockComponent("Causal"), # type: ignore
        symbol_grounding=MockComponent("SymbolGrounding"), # type: ignore
        
        # Inject Sleep Modules
        astrocyte_network=astrocyte,
        sleep_consolidator=sleep_manager,
        device=device
    )
    
    return brain

def simulate_daytime_experiences(brain: ArtificialBrain, num_experiences: int = 5):
    """
    日中の活動をシミュレーションし、メモリに「成功体験」を書き込む。
    """
    logger.info(f"☀️ Simulating daytime: Experiencing {num_experiences} tasks...")
    
    # ダミーの入力データ (本来はTokenizerで変換済みID)
    vocab_size = 50300
    seq_len = 16
    
    # SleepConsolidatorへのアクセス用
    if brain.sleep_manager is None:
        logger.error("Sleep manager not initialized!")
        return

    # SleepConsolidator -> Memory へのアクセス
    memory_system = brain.sleep_manager.memory

    for i in range(num_experiences):
        # ランダムな状況と、それに対する「正解行動（Thought Trace）」を生成
        input_ids = torch.randint(0, vocab_size, (1, seq_len)).tolist()
        
        # Fix: 型ヒントを明示して mypy エラー (object vs specific types) を回避
        experience: Dict[str, Any] = {
            "state": {"context": f"Problem {i}"},
            "action": "solved_problem",
            "result": "Success",
            "reward": {"external": 1.0 + random.random()}, # High reward!
            "expert_used": ["reasoning_engine"],
            "decision_context": {"reason": "Logic verified"},
            
            # SleepConsolidatorが学習に使うデータ
            "encoded_input": input_ids,
            "action_id": 1 # Dummy action ID
        }
        
        # Memoryに記録 (Any型を含む辞書から値を取り出すため、mypyはこれを受け入れるようになる)
        memory_system.record_experience(
            state=experience["state"],
            action=experience["action"],
            result=experience["result"],
            reward=experience["reward"],
            expert_used=experience["expert_used"],
            decision_context=experience["decision_context"]
        )
        
        time.sleep(0.1)

    logger.info("📝 Experiences recorded in Hippocampus (Short-term Memory).")

def main():
    logger.info("============================================================")
    logger.info("🌙 SNN Sleep & Consolidation Demo")
    logger.info("============================================================")
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info("🚀 Using CUDA GPU")

    # 1. 脳の構築 (疲れた状態で初期化)
    brain = build_brain_with_sleep(device)
    
    initial_health = brain.get_brain_status()
    # 辞書アクセスを安全に行う
    if 'astrocyte' in initial_health and 'metrics' in initial_health['astrocyte']:
        metrics = initial_health['astrocyte']['metrics']
        
        # [Fix] KeyError対策: get() を使用し、欠損時はデフォルト値または安全なフォールバック
        energy = metrics.get('energy_percent', 0.0)
        fatigue = metrics.get('fatigue_index', 0.0)
        
        logger.info(f"😫 Initial State: Energy={energy:.1f}%, Fatigue={fatigue:.1f}")
        
        if 'energy_percent' not in metrics:
            logger.warning(f"⚠️ 'energy_percent' missing in metrics. Available keys: {list(metrics.keys())}")
    
    # 2. 日中の活動 (経験の蓄積)
    simulate_daytime_experiences(brain, num_experiences=10)
    
    # 3. 睡眠サイクルの実行
    logger.info("\n🛌 Initiating Sleep Cycle (8 hours)...")
    start_time = time.time()
    
    # ArtificialBrain.sleep_cycle() を呼び出す
    # 内部で Astrocyteの回復 と SleepConsolidatorの学習 が走る
    brain.sleep_cycle()
    
    duration = time.time() - start_time
    logger.info(f"⏰ Sleep process took {duration:.3f}s (Simulated 8h)")

    # 4. 覚醒後の状態確認
    logger.info("\n🌅 Waking up...")
    final_health = brain.get_brain_status()
    
    if 'astrocyte' in final_health and 'metrics' in final_health['astrocyte']:
        metrics_new = final_health['astrocyte']['metrics']
        
        energy_new = metrics_new.get('energy_percent', 100.0)
        fatigue_new = metrics_new.get('fatigue_index', 0.0)
        
        logger.info(f"🤩 Final State: Energy={energy_new:.1f}%, Fatigue={fatigue_new:.1f}")
        
        # 検証
        if energy_new > 90.0 and fatigue_new < 10.0:
            logger.info("✅ Homeostasis Check: PASSED (Fully recovered)")
        else:
            logger.error("❌ Homeostasis Check: FAILED")
    else:
        logger.error("❌ Health status format invalid.")

    # 夢の効果確認 (ログから判断)
    logger.info("============================================================")
    logger.info("🎉 Sleep Cycle Demo Completed.")

if __name__ == "__main__":
    main()