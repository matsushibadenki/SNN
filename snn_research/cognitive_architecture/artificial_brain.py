# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# (修正: sleep_and_dream の NoneType エラーを修正)
#
# Title: 人工脳 統合認知サイクル
# Description:
# - sleep_and_dream 内で emotion が None の場合にクラッシュする問題を修正。
# - Workspace からのデータ取得時に型チェックを強化。

from typing import Dict, Any, List, Optional, Callable, cast
import asyncio
import re
import torch
import numpy as np
import random

# IO and encoding
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
# Core cognitive modules
from .hybrid_perception_cortex import HybridPerceptionCortex
from .visual_perception import VisualCortex
from .prefrontal_cortex import PrefrontalCortex
# Memory systems
from .hippocampus import Hippocampus
from .cortex import Cortex
# Value and action selection
from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
# Motor control
from .cerebellum import Cerebellum
from .motor_cortex import MotorCortex
# Central hub
from .global_workspace import GlobalWorkspace
# Causal Engine
from .causal_inference_engine import CausalInferenceEngine
# Motivation System
from .intrinsic_motivation import IntrinsicMotivationSystem
# Symbol Grounding
from .symbol_grounding import SymbolGrounding
# Utils
from torchvision import transforms # type: ignore


class ArtificialBrain:
    """
    認知アーキテクチャ全体を統合し、「意識的認知サイクル」を制御する人工脳システム。
    """
    image_transform: Callable[[Any], torch.Tensor]

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
        symbol_grounding: SymbolGrounding
    ):
        print("🚀 人工脳システムの起動を開始...")
        self.workspace = global_workspace
        self.motivation_system = motivation_system
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator
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
        
        self.cycle_count = 0
        self.is_sleeping = False
        
        self.image_transform = cast(Callable[[Any], torch.Tensor], transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]))
        
        print("✅ 人工脳システムの全モジュールが正常に起動しました。")

    def run_cognitive_cycle(self, raw_input: Any):
        if self.is_sleeping:
            print("💤 人工脳は睡眠中です。入力を無視します。")
            return

        self.cycle_count += 1
        input_preview = str(raw_input)[:50] + "..." if isinstance(raw_input, str) else type(raw_input).__name__
        print(f"\n--- 🧠 新しい認知サイクルを開始 ({self.cycle_count}) --- \n入力: {input_preview}")
        
        # 1. 感覚入力
        sensory_info = self.receptor.receive(raw_input)
        perception_context = "unknown"

        if sensory_info['type'] == 'image':
            try:
                transform_fn = self.image_transform
                transformed_img = transform_fn(sensory_info['content'])
                image_tensor = transformed_img.unsqueeze(0)
                self.visual_cortex.perceive_and_upload(image_tensor)
                perception_context = "visual_input"
            except Exception as e:
                print(f"❌ 視覚処理エラー: {e}")
        else:
            spike_pattern = self.encoder.encode(sensory_info, duration=50) 
            self.perception.perceive_and_upload(spike_pattern)
            if sensory_info['type'] == 'text':
                self.amygdala.evaluate_and_upload(sensory_info['content'])
            perception_context = f"text_input_{str(raw_input)[:10]}"

        # 1c. 記号接地と記憶検索
        target_info = self.workspace.get_information("visual_cortex") or self.workspace.get_information("perception")
        if target_info and isinstance(target_info, dict):
            features = target_info.get('features')
            if features is not None and isinstance(features, torch.Tensor):
                self.symbol_grounding.ground_neural_pattern(features, perception_context)
                self.hippocampus.evaluate_relevance_and_upload(features)

        # 2. 意識のブロードキャスト
        self.workspace.conscious_broadcast_cycle()
        conscious_content = self.workspace.conscious_broadcast_content
        
        if conscious_content is not None:
            # 3-4. 行動選択と実行 (各モジュールがWorkspaceを購読して反応)
            selected_action = self.basal_ganglia.selected_action
            if selected_action:
                motor_commands = self.cerebellum.refine_action_plan(selected_action)
                command_logs = self.motor.execute_commands(motor_commands)
                self.actuator.run_command_sequence(command_logs)
                self.symbol_grounding.process_observation(selected_action, "action_execution")

            # 5. 記憶の保存
            amygdala_info = self.workspace.get_information("amygdala")
            episode = {
                'type': 'conscious_experience', 
                'content': conscious_content, 
                'source_input': str(raw_input),
                'emotion': amygdala_info if amygdala_info else {} # Noneの場合は空辞書
            }
            self.hippocampus.store_episode(episode)

            # 6. フィードバック
            prediction_error = 0.9 if self.causal_engine.just_inferred else 0.1
            self.causal_engine.reset_inference_flag()
            success_rate = 1.0 if selected_action else 0.0
            loss = 0.1
        else:
            print("🤔 意識に上る情報がなく、サイクルをスキップしました。")
            prediction_error, success_rate, loss = 0.1, 0.0, 0.1

        self.motivation_system.update_metrics(prediction_error, success_rate, 0.0, loss)

        # 7. 睡眠サイクル
        if self.cycle_count % 20 == 0:
            self.sleep_and_dream()

        print("--- ✅ 認知サイクル完了 ---")

    def sleep_and_dream(self):
        print("\n💤 --- 睡眠モード開始 (Memory Consolidation & Dreaming) ---")
        self.is_sleeping = True
        
        recent_memories = list(self.hippocampus.working_memory)
        if recent_memories:
            print(f"  💭 {len(recent_memories)} 個のエピソードから夢を生成中...")
            dream_sequence = random.sample(recent_memories, min(len(recent_memories), 5))
            
            for episode in dream_sequence:
                # --- ▼ 修正: Noneチェックの強化 ▼ ---
                emotion = episode.get('emotion')
                # emotion が None の場合（get_informationがNoneを返した場合）の対策
                if emotion is None:
                    emotion = {}
                
                # 辞書型であることを確認してからアクセス
                valence = 0.0
                arousal = 0.0
                if isinstance(emotion, dict):
                    valence = float(emotion.get('valence', 0.0))
                    arousal = float(emotion.get('arousal', 0.0))
                # --- ▲ 修正 ▲ ---
                
                importance = abs(valence) + arousal
                content_preview = str(episode.get('content', ''))[:30]
                print(f"  📽️ リプレイ中: '{content_preview}...' (重要度: {importance:.2f})")
                
                if importance > 0.5:
                    self.cortex.consolidate_memory(episode)
                    print("    --> 🧠 大脳皮質へ固定化されました。")
        
        self.hippocampus.clear_memory()
        self.is_sleeping = False
        print("🌅 --- 目覚め (Ready for new cycles) ---\n")

    def consolidate_memories(self):
        episodes = self.hippocampus.get_and_clear_episodes_for_consolidation()
        for episode in episodes:
            self.cortex.consolidate_memory(episode)

    def correct_knowledge(self, concept: str, new_info: str, reason: str = "user correction"):
        print(f"\n🛠️ 知識修正: '{concept}' -> '{new_info}'")
        correction_episode = {
            'type': 'knowledge_correction', 'concept': concept,
            'content': {'text': new_info}, 'source': 'user', 'reason': reason
        }
        self.hippocampus.store_episode(correction_episode)
        if hasattr(self.cortex, 'rag_system') and self.cortex.rag_system:
             self.cortex.rag_system.add_relationship(concept, "is_corrected_to", new_info)