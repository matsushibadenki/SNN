# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# ファイル名: 人工脳統合コア
# 機能説明: 全ての認知モジュール（知覚、記憶、感情、制御）を統合し、
#          「意識的認知サイクル (Conscious Cognitive Cycle)」を実行するクラス。
#          外部入力の処理、意識のブロードキャスト、行動選択、記憶の固定化（睡眠）を管理する。

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
        
        # 画像変換パイプライン
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
        
        # 1. 感覚入力 (Reception & Perception)
        sensory_info = self.receptor.receive(raw_input)
        perception_context = "unknown"

        if sensory_info['type'] == 'image':
            try:
                transformed_img = self.image_transform(sensory_info['content'])
                image_tensor = transformed_img.unsqueeze(0)
                self.visual_cortex.perceive_and_upload(image_tensor)
                perception_context = "visual_input"
            except Exception as e:
                print(f"❌ 視覚処理エラー: {e}")
        else:
            # テキストまたは数値をスパイク化して知覚野へ
            duration = 50 # ミリ秒相当
            spike_pattern = self.encoder.encode(sensory_info, duration=duration) 
            self.perception.perceive_and_upload(spike_pattern)
            
            # テキストの場合は扁桃体で情動評価も行う
            if sensory_info['type'] == 'text':
                self.amygdala.evaluate_and_upload(str(sensory_info['content']))
            perception_context = f"text_input_{str(raw_input)[:10]}"

        # 1c. 記号接地と記憶検索
        target_info = self.workspace.get_information("visual_cortex") or self.workspace.get_information("perception")
        if target_info and isinstance(target_info, dict):
            features = target_info.get('features')
            if features is not None and isinstance(features, torch.Tensor):
                self.symbol_grounding.ground_neural_pattern(features, perception_context)
                self.hippocampus.evaluate_relevance_and_upload(features)

        # 2. 意識のブロードキャスト (Conscious Access)
        # 各モジュールからの情報を競合させ、最も顕著な情報を全体に共有する
        self.workspace.conscious_broadcast_cycle()
        conscious_content = self.workspace.conscious_broadcast_content
        
        if conscious_content is not None:
            # 3. 意思決定と行動計画 (PFC & Basal Ganglia)
            # 前頭前野は目標を更新し、大脳基底核は具体的な行動を選択する
            # (これらはWorkspaceを購読しているため、broadcast時に内部状態を更新済み)
            
            selected_action = self.basal_ganglia.selected_action
            
            # 4. 行動実行 (Action Execution)
            if selected_action:
                motor_commands = self.cerebellum.refine_action_plan(selected_action)
                command_logs = self.motor.execute_commands(motor_commands)
                self.actuator.run_command_sequence(command_logs)
                # 自分の行動を観測としてフィードバック
                self.symbol_grounding.process_observation(selected_action, "action_execution")

            # 5. エピソード記憶の保存
            amygdala_info = self.workspace.get_information("amygdala")
            episode = {
                'type': 'conscious_experience', 
                'content': conscious_content, 
                'source_input': str(raw_input),
                'emotion': amygdala_info if amygdala_info else {} # 安全に空辞書を渡す
            }
            self.hippocampus.store_episode(episode)

            # 6. フィードバックと学習 (因果推論・動機付け)
            prediction_error = 0.9 if self.causal_engine.just_inferred else 0.1
            self.causal_engine.reset_inference_flag()
            success_rate = 1.0 if selected_action else 0.0
            loss = 0.1 # 簡易的な損失値
        else:
            print("🤔 意識に上る情報がなく、サイクルをスキップしました。")
            prediction_error, success_rate, loss = 0.1, 0.0, 0.1

        self.motivation_system.update_metrics(prediction_error, success_rate, 0.0, loss)

        # 7. 睡眠サイクル (一定周期で記憶を整理)
        if self.cycle_count % 20 == 0:
            self.sleep_and_dream()

        print("--- ✅ 認知サイクル完了 ---")

    def sleep_and_dream(self):
        """
        睡眠フェーズ。海馬（短期記憶）から大脳皮質（長期記憶）への記憶の固定化（Consolidation）を行い、
        夢（ランダムな記憶の再生）を通じて重要な情報を強化する。
        """
        print("\n💤 --- 睡眠モード開始 (Memory Consolidation & Dreaming) ---")
        self.is_sleeping = True
        
        recent_memories = list(self.hippocampus.working_memory)
        if recent_memories:
            print(f"  💭 {len(recent_memories)} 個のエピソードから夢を生成中...")
            # 夢を見る（ランダムサンプリング）
            dream_sequence = random.sample(recent_memories, min(len(recent_memories), 5))
            
            for episode in dream_sequence:
                # 感情価に基づいて記憶の重要度を判定
                emotion = episode.get('emotion')
                # Noneチェックと型安全なアクセス
                if emotion is None:
                    emotion = {}
                
                valence = 0.0
                arousal = 0.0
                if isinstance(emotion, dict):
                    valence = float(emotion.get('valence', 0.0))
                    arousal = float(emotion.get('arousal', 0.0))
                
                importance = abs(valence) + arousal
                content_preview = str(episode.get('content', ''))[:30]
                
                print(f"  📽️ リプレイ中: '{content_preview}...' (重要度: {importance:.2f})")
                
                # 重要な記憶のみを長期記憶へ固定化
                if importance > 0.5:
                    self.cortex.consolidate_memory(episode)
                    print("    --> 🧠 大脳皮質へ固定化されました。")
        
        # 短期記憶をクリア
        self.hippocampus.clear_memory()
        self.is_sleeping = False
        print("🌅 --- 目覚め (Ready for new cycles) ---\n")

    def consolidate_memories(self):
        """手動で記憶の固定化を行うユーティリティ"""
        episodes = self.hippocampus.get_and_clear_episodes_for_consolidation()
        for episode in episodes:
            self.cortex.consolidate_memory(episode)

    def correct_knowledge(self, concept: str, new_info: str, reason: str = "user correction"):
        """
        ユーザーからの指示に基づいて知識を修正する。
        """
        print(f"\n🛠️ 知識修正: '{concept}' -> '{new_info}'")
        correction_episode = {
            'type': 'knowledge_correction', 'concept': concept,
            'content': {'text': new_info}, 'source': 'user', 'reason': reason
        }
        self.hippocampus.store_episode(correction_episode)
        # 即座に反映
        if hasattr(self.cortex, 'rag_system') and self.cortex.rag_system:
             self.cortex.rag_system.add_relationship(concept, "is_corrected_to", new_info)
