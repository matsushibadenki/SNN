# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# Title: Artificial Brain Kernel v14.0
# Description:
#   人工脳のライフサイクル（覚醒・睡眠・進化）を制御するカーネル。
#   - 状態管理: Awake, Sleeping, Dreaming
#   - 認知サイクル: 感覚 -> 記号接地 -> 意識 -> 意思決定 -> 行動
#   - 恒常性: エネルギー（グルコース）管理と疲労蓄積による睡眠誘導

from typing import Dict, Any, List, Optional, Callable, cast
import asyncio
import torch
import time
import logging

# 各コンポーネントのインポート
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
from .visual_perception import VisualCortex
from .prefrontal_cortex import PrefrontalCortex
from .hippocampus import Hippocampus
from .cortex import Cortex
from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
from .cerebellum import Cerebellum
from .motor_cortex import MotorCortex
from .global_workspace import GlobalWorkspace
from .causal_inference_engine import CausalInferenceEngine
from .intrinsic_motivation import IntrinsicMotivationSystem
from .symbol_grounding import SymbolGrounding
from .sleep_consolidation import SleepConsolidator
from .hybrid_perception_cortex import HybridPerceptionCortex

from torchvision import transforms # type: ignore

logger = logging.getLogger(__name__)

class ArtificialBrain:
    """
    ニューロシンボリック・人工脳システム (v14.0)
    """
    def __init__(
        self,
        global_workspace: GlobalWorkspace,
        motivation_system: IntrinsicMotivationSystem,
        sensory_receptor: SensoryReceptor,
        spike_encoder: SpikeEncoder,
        actuator: Actuator,
        perception_cortex: HybridPerceptionCortex,
        visual_cortex: VisualCortex,
        prefrontal_cortex: PrefrontalCortex,
        hippocampus: Hippocampus,
        cortex: Cortex,
        amygdala: Amygdala,
        basal_ganglia: BasalGanglia,
        cerebellum: Cerebellum,
        motor_cortex: MotorCortex,
        causal_inference_engine: CausalInferenceEngine,
        symbol_grounding: SymbolGrounding,
        sleep_consolidator: Optional[SleepConsolidator] = None
    ):
        logger.info("🚀 Initializing Artificial Brain Kernel v14.0...")
        
        self.workspace = global_workspace
        self.motivation_system = motivation_system
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator
        
        # Cognitive Modules
        self.perception = perception_cortex
        self.visual_cortex = visual_cortex
        self.pfc = prefrontal_cortex
        self.hippocampus = hippocampus
        self.cortex = cortex
        self.amygdala = amygdala
        self.basal_ganglia = basal_ganglia
        self.cerebellum = cerebellum
        self.motor = motor_cortex
        self.causal_engine = causal_inference_engine
        self.symbol_grounding = symbol_grounding
        self.sleep_consolidator = sleep_consolidator
        
        # State Variables
        self.cycle_count = 0
        self.is_sleeping = False
        self.energy_level = 100.0 # 仮想グルコースレベル
        self.fatigue_level = 0.0  # 疲労度 (睡眠圧)
        
        # 画像変換
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        logger.info("✅ Artificial Brain Kernel ready.")

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        1回の認知サイクルを実行する (覚醒時)。
        """
        # 1. 状態チェック (睡眠・エネルギー)
        if self.is_sleeping:
            logger.info("💤 Brain is sleeping. Input ignored.")
            return {"status": "sleeping"}
            
        if self.energy_level <= 0:
            logger.warning("🪫 Energy depleted. Forced sleep initiated.")
            self.sleep_and_dream()
            return {"status": "forced_sleep"}

        self.cycle_count += 1
        self.energy_level -= 0.5 # 基礎代謝
        self.fatigue_level += 1.0
        
        cycle_report = {"cycle": self.cycle_count, "input": str(raw_input)[:30]}
        # ログ出力フォーマットを調整
        # print(f"\n--- 🧠 Cognitive Cycle #{self.cycle_count} (Energy: {self.energy_level:.1f}%) ---")

        # 2. 知覚 (Perception & Grounding)
        sensory_info = self.receptor.receive(raw_input)
        
        if sensory_info['type'] == 'image':
            # 視覚処理
            try:
                img_tensor = self.image_transform(sensory_info['content']).unsqueeze(0)
                self.visual_cortex.perceive_and_upload(img_tensor)
                # 視覚特徴の接地
                vis_data = self.workspace.get_information("visual_cortex")
                if vis_data and "features" in vis_data:
                    self.symbol_grounding.ground_neural_pattern(vis_data["features"], "visual_input")
            except Exception as e:
                logger.error(f"Visual processing error: {e}")
        else:
            # 言語/一般処理
            content_str = str(sensory_info['content'])
            spike_pattern = self.encoder.encode(sensory_info, duration=16)
            self.perception.perceive_and_upload(spike_pattern)
            
            # 情動評価
            self.amygdala.evaluate_and_upload(content_str)
            
            # 言語情報の接地（観測データとして）
            self.symbol_grounding.process_observation({"text": content_str}, "text_input")

        # 3. 意識のワークスペース競合 (Global Workspace Theory)
        self.workspace.conscious_broadcast_cycle()
        conscious_content = self.workspace.conscious_broadcast_content
        
        cycle_report["conscious_content"] = str(conscious_content)[:50] if conscious_content else "None"

        # 4. 意思決定と実行 (Action)
        if conscious_content:
            # 前頭前野による目標更新 (意識的内容を受けて更新)
            # ログの順序に合わせて、Broadcast後に処理を実行
            self.pfc.handle_conscious_broadcast("workspace", conscious_content)
            
            # 大脳基底核による行動選択
            # (PFCの目標もWorkspace経由でBasalGangliaに伝わる前提)
            selected_action = self.basal_ganglia.selected_action 
            
            if selected_action:
                action_name = selected_action.get('action')
                cycle_report["action"] = action_name
                # logger.info(f"⚡ Action Selected: {action_name}")
                
                # 小脳による運動計画
                motor_commands = self.cerebellum.refine_action_plan(selected_action)
                # 運動野による実行
                logs = self.motor.execute_commands(motor_commands)
                # アクチュエータ出力
                self.actuator.run_command_sequence(logs)
                
                # 行動によるエネルギー消費
                self.energy_level -= 2.0
            
            # 5. 記憶の形成 (Hippocampus)
            # エピソード記憶を作成
            amygdala_info = self.workspace.get_information("amygdala")
            episode = {
                "timestamp": time.time(),
                "input": str(raw_input),
                "consciousness": conscious_content,
                "action": selected_action,
                "emotion": amygdala_info
            }
            self.hippocampus.store_episode(episode)
            
            # 6. 因果推論の更新
            # (CausalInferenceEngineはWorkspaceを購読しており自動更新される)

        # 7. 睡眠圧のチェック
        if self.fatigue_level > 50 or (self.hippocampus.working_memory and len(self.hippocampus.working_memory) >= self.hippocampus.capacity):
            logger.info("🥱 Fatigue high. Initiating sleep cycle...")
            self.sleep_and_dream()
            
        return cycle_report

    def sleep_and_dream(self):
        """
        睡眠フェーズを実行する。
        """
        self.is_sleeping = True
        print(f"\n💤 --- SLEEP MODE ACTIVATED (Fatigue: {self.fatigue_level}) ---")
        
        # 1. 記憶の固定化 (Explicit)
        # 海馬から重要なエピソードを取り出し、長期記憶(GraphRAG)へ
        episodes = self.hippocampus.get_and_clear_episodes_for_consolidation()
        print(f"  📝 Consolidating {len(episodes)} episodes to Cortex...")
        for ep in episodes:
            self.cortex.consolidate_memory(ep)
            
        # 2. ニューラルリプレイ (Implicit)
        if self.sleep_consolidator:
            report = self.sleep_consolidator.perform_sleep_cycle()
            print(f"  🧠 Replay finished. Synaptic change: {report.get('synaptic_change', 0):.4f}")
        else:
            logger.warning("  ⚠️ SleepConsolidator not attached. Skipping neural replay.")
            time.sleep(1) # 簡易的な休息
            
        # 3. リフレッシュ
        self.fatigue_level = 0.0
        self.energy_level = 100.0
        self.is_sleeping = False
        print("🌅 --- WAKE UP --- \n")

    def user_correction(self, concept: str, correct_info: str):
        """ユーザーによる知識の訂正を受け付ける"""
        logger.info(f"🛠️ User Correction: {concept} -> {correct_info}")
        # Cortexを通じて知識グラフを修正
        if hasattr(self.cortex, 'rag_system') and self.cortex.rag_system:
             self.cortex.rag_system.update_knowledge(concept, "is_corrected_to", correct_info, reason="user_instruction")
