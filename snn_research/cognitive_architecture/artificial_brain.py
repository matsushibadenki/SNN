# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# Title: Artificial Brain Kernel v14.0 (Neuro-Symbolic OS Edition)
# Description:
#   ロードマップ Phase 5 & 7 に基づく、人工脳のオペレーティングシステム実装。
#   認知モジュールのオーケストレーション、エネルギー恒常性維持、
#   および覚醒-睡眠サイクル（神経-記号還流）を自律的に制御する。

from typing import Dict, Any, List, Optional, Union
import time
import logging
import torch
from torchvision import transforms # type: ignore

# Core Modules
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

# Cognitive Modules
from .visual_perception import VisualCortex
from .hybrid_perception_cortex import HybridPerceptionCortex
from .prefrontal_cortex import PrefrontalCortex
from .hippocampus import Hippocampus
from .cortex import Cortex
from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
from .cerebellum import Cerebellum
from .motor_cortex import MotorCortex
from .causal_inference_engine import CausalInferenceEngine
from .intrinsic_motivation import IntrinsicMotivationSystem
from .symbol_grounding import SymbolGrounding
from .sleep_consolidation import SleepConsolidator

logger = logging.getLogger(__name__)

class ArtificialBrain:
    """
    Artificial Brain Kernel v14.0
    自律的にエネルギーを管理し、学習（覚醒）と整理（睡眠）を繰り返すニューロシンボリックOS。
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
        sleep_consolidator: Optional[SleepConsolidator] = None,
        astrocyte_network: Optional[AstrocyteNetwork] = None
    ):
        logger.info("🚀 Booting Artificial Brain Kernel v14.0...")
        
        # --- Kernel Core ---
        self.workspace = global_workspace
        self.astrocyte = astrocyte_network # ニューロモルフィックOSのスケジューラ
        self.motivation = motivation_system
        self.sleep_manager = sleep_consolidator
        
        # --- IO Interfaces ---
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator
        
        # --- Cognitive Modules ---
        self.perception = perception_cortex
        self.visual = visual_cortex
        self.pfc = prefrontal_cortex
        self.hippocampus = hippocampus
        self.cortex = cortex
        self.amygdala = amygdala
        self.basal_ganglia = basal_ganglia
        self.cerebellum = cerebellum
        self.motor = motor_cortex
        self.causal_engine = causal_inference_engine
        self.grounding = symbol_grounding
        
        # --- System State (Homeostasis) ---
        self.cycle_count = 0
        self.state = "AWAKE" # AWAKE, SLEEPING, DREAMING
        self.energy_level = 100.0 # グルコースレベル (%)
        self.fatigue_level = 0.0  # 疲労度 (睡眠圧)
        
        # Utilities
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        logger.info("✅ Artificial Brain System initialized successfully.")

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        1回の認知サイクルを実行する（覚醒時）。
        感覚 -> 知覚 -> 意識 -> 決定 -> 行動 -> 記憶 -> 恒常性調整
        """
        start_time = time.time()
        
        # 1. OS状態チェック (睡眠・エネルギー)
        if self.state in ["SLEEPING", "DREAMING"]:
            logger.info("💤 Brain is sleeping... (Input buffered or ignored)")
            return {"status": "sleeping", "response": "Zzz..."}
            
        if self.energy_level <= 5.0 or self.fatigue_level >= 90.0:
            logger.warning("⚠️ Critical Fatigue/Energy levels. Initiating forced sleep.")
            return self.sleep_cycle()

        self.cycle_count += 1
        # 基礎代謝コスト
        self._consume_energy(0.5)
        self.fatigue_level += 0.5
        
        report: Dict[str, Any] = {"cycle": self.cycle_count, "input": str(raw_input)[:50]}

        # --- Step 1: Perception & Grounding (入力処理) ---
        sensory_info = self.receptor.receive(raw_input)
        
        if sensory_info['type'] == 'image':
            # 視覚経路 (Ventral/Dorsal streams)
            img_tensor = self.image_transform(sensory_info['content']).unsqueeze(0)
            self.visual.perceive_and_upload(img_tensor)
            
            # 視覚的シンボル接地 (Neural pattern -> Symbol)
            vis_data = self.workspace.get_information("visual_cortex")
            if vis_data and "features" in vis_data:
                concept_id = self.grounding.ground_neural_pattern(vis_data["features"], "visual_input")
                report["grounded_concept"] = concept_id
        else:
            # 言語/テキスト経路
            content_str = str(sensory_info['content'])
            # 意味的スパイクエンコーディング
            spike_pattern = self.encoder.encode(sensory_info, duration=16)
            self.perception.perceive_and_upload(spike_pattern)
            
            # 情動評価 (Valence/Arousal)
            self.amygdala.evaluate_and_upload(content_str)
            
            # 観測データのシンボル化
            self.grounding.process_observation({"text": content_str}, "text_input")

        # --- Step 2: Consciousness (意識の競合とブロードキャスト) ---
        # GWT (Global Workspace Theory) に基づく情報統合
        self.workspace.conscious_broadcast_cycle()
        conscious_content = self.workspace.conscious_broadcast_content
        report["consciousness"] = str(conscious_content)[:50] if conscious_content else None

        # --- Step 3: Decision Making & Action (意思決定) ---
        if conscious_content:
            # 前頭前野 (PFC): 文脈に基づく目標(Goal)の更新
            self.pfc.handle_conscious_broadcast("workspace", conscious_content)
            
            # 大脳基底核 (BG): アクションの選択 (Go/No-Go)
            # 扁桃体の情動価によって閾値が変調される
            amygdala_state = self.workspace.get_information("amygdala")
            selected_action = self.basal_ganglia.select_action(
                # ここでは簡易的に候補を生成（本来はPlanner等から来る）
                action_candidates=[
                    {'action': 'reply_text', 'value': 0.8}, 
                    {'action': 'store_memory', 'value': 0.6},
                    {'action': 'ignore', 'value': 0.1}
                ],
                emotion_context=amygdala_state
            )
            
            if selected_action:
                action_name = selected_action.get('action')
                report["action"] = action_name
                
                # 小脳 & 運動野: 行動の具体化と実行
                motor_commands = self.cerebellum.refine_action_plan(selected_action)
                execution_log = self.motor.execute_commands(motor_commands)
                self.actuator.run_command_sequence(execution_log)
                
                # 行動コスト
                self._consume_energy(2.0)
            
            # --- Step 4: Memory Formation (エピソード記憶) ---
            # 海馬への一時保管
            episode = {
                "timestamp": time.time(),
                "input": str(raw_input),
                "consciousness": conscious_content,
                "action": selected_action,
                "emotion": amygdala_state
            }
            self.hippocampus.store_episode(episode)
            
            # 6. 因果推論の更新
            # (CausalInferenceEngineはWorkspaceを購読しており自動更新される)

        # --- Step 5: Homeostasis (恒常性維持) ---
        if self.astrocyte:
            # アストロサイトによる神経活動の調整（メタ可塑性）
            self.astrocyte.step()

        # 疲労判定
        self._check_fatigue()
        
        report["energy"] = self.energy_level
        report["fatigue"] = self.fatigue_level
        
        return report

    def sleep_cycle(self) -> Dict[str, Any]:
        """
        睡眠サイクルを実行する。
        Phase 5: Neuro-Symbolic Feedback Loop の核心。
        GraphRAG(記号)からSNN(神経)への情報の逆流（Replay）を行う。
        """
        self.state = "SLEEPING"
        logger.info("\n🌙 --- SLEEP CYCLE INITIATED ---")
        logger.info(f"   Initial Fatigue: {self.fatigue_level:.1f}, Energy: {self.energy_level:.1f}")

        report = {"status": "slept", "phases": []}

        # Phase 1: Explicit Consolidation (NREM sleep like)
        # 海馬（短期記憶）-> 大脳皮質（長期記憶/GraphRAG）への転送
        episodes = self.hippocampus.get_and_clear_episodes_for_consolidation()
        if episodes:
            logger.info(f"   📝 Consolidating {len(episodes)} episodes to Cortex (GraphRAG)...")
            for ep in episodes:
                self.cortex.consolidate_memory(ep)
            report["phases"].append(f"Consolidated {len(episodes)} episodes")
        
        # Phase 2: Implicit Consolidation / Dreaming (REM sleep like)
        # GraphRAGからの知識再生によるニューラルネットワークの微調整
        if self.sleep_manager:
            self.state = "DREAMING"
            dream_stats = self.sleep_manager.perform_sleep_cycle()
            synaptic_change = dream_stats.get('synaptic_change', 0.0)
            dreams_count = dream_stats.get('dreams_replayed', 0)
            
            logger.info(f"   🦄 Dream Cycle: Replayed {dreams_count} concepts.")
            logger.info(f"   🧠 Neural Plasticity: Total synaptic weight change: {synaptic_change:.4f}")
            report["phases"].append(f"Dreamed {dreams_count} concepts (DeltaW: {synaptic_change:.4f})")
            
            # 夢の内容に基づく新たな洞察（仮）
            if synaptic_change > 1.0:
                logger.info("   💡 Insight: Significant neural reorganization occurred.")
        else:
            logger.warning("   ⚠️ SleepManager (Consolidator) not attached. Skipping dream phase.")
        
        # Phase 3: Restoration
        self.fatigue_level = max(0.0, self.fatigue_level - 80.0)
        self.energy_level = 100.0 # 完全回復
        self.state = "AWAKE"
        
        logger.info("🌅 --- WAKE UP ---")
        logger.info(f"   Final Fatigue: {self.fatigue_level:.1f}, Energy: {self.energy_level:.1f}\n")
        
        return report

    def _consume_energy(self, amount: float):
        """エネルギー消費と枯渇時のペナルティ"""
        self.energy_level = max(0.0, self.energy_level - amount)
        if self.energy_level < 20.0:
            # エネルギー不足時は疲労が加速する
            self.fatigue_level += amount * 2.0

    def _check_fatigue(self):
        """疲労度チェックと自動睡眠トリガー"""
        memory_load = len(self.hippocampus.working_memory) / self.hippocampus.capacity
        
        # 疲労が高い、またはワーキングメモリが一杯になったら睡眠
        if self.fatigue_level > 80.0 or memory_load >= 1.0:
            logger.info("🥱 Drowsiness detected. Initiating sleep cycle...")
            # 自動睡眠
            self.sleep_cycle()
    
    def correct_knowledge(self, concept: str, correct_info: str, reason: str = "user_correction"):
        """
        ユーザーからの明示的な知識修正を受け付けるインターフェース。
        即座にGraphRAGを更新し、次回の睡眠サイクルでのSNN修正を予約する。
        """
        logger.info(f"🛠️ Knowledge Correction Request: '{concept}' -> '{correct_info}'")
        
        # 1. 知識グラフの更新 (Symbolic Update)
        if self.cortex.rag_system:
            self.cortex.rag_system.update_knowledge(
                subj=concept, 
                pred="is_corrected_to", 
                new_obj=correct_info, 
                reason=reason
            )
            
        # 2. 短期記憶への修正イベントの注入 (Neural Update Trigger)
        # これにより、次回の睡眠時にこの修正情報が高い優先度でリプレイされる
        correction_episode = {
            "type": "knowledge_correction",
            "concept": concept,
            "content": correct_info,
            "reason": reason,
            "timestamp": time.time()
        }
        self.hippocampus.store_episode(correction_episode)
        logger.info("   -> Correction stored in Hippocampus for sleep consolidation.")
