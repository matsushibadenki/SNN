# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (Fix: PFC Alias Added)
# 目的: テストコードとの互換性のため、prefrontal_cortex のエイリアス pfc を追加。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, cast
import logging

from .global_workspace import GlobalWorkspace
from .hippocampus import Hippocampus
from .cortex import Cortex
from .basal_ganglia import BasalGanglia
from .motor_cortex import MotorCortex
from .amygdala import Amygdala
from .prefrontal_cortex import PrefrontalCortex
from .perception_cortex import PerceptionCortex
from .intrinsic_motivation import IntrinsicMotivationSystem
from .astrocyte_network import AstrocyteNetwork
from .reasoning_engine import ReasoningEngine
from .meta_cognitive_snn import MetaCognitiveSNN
from snn_research.modules.reflex_module import ReflexModule
from snn_research.models.experimental.world_model_snn import SpikingWorldModel
from snn_research.safety.ethical_guardrail import EthicalGuardrail
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator

logger = logging.getLogger(__name__)

class ArtificialBrain(nn.Module):
    """
    複数の脳領域モジュールを統合する人工脳メインクラス。
    外部からコンポーネントを注入することを基本とする。
    """
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        # 依存性の注入 (Optional)
        global_workspace: Optional[GlobalWorkspace] = None,
        astrocyte_network: Optional[AstrocyteNetwork] = None,
        motivation_system: Optional[IntrinsicMotivationSystem] = None,
        sensory_receptor: Optional[SensoryReceptor] = None,
        spike_encoder: Optional[SpikeEncoder] = None,
        actuator: Optional[Actuator] = None,
        thinking_engine: Optional[nn.Module] = None,
        perception_cortex: Optional[PerceptionCortex] = None,
        visual_cortex: Optional[Any] = None,
        prefrontal_cortex: Optional[PrefrontalCortex] = None,
        hippocampus: Optional[Hippocampus] = None,
        cortex: Optional[Cortex] = None,
        amygdala: Optional[Amygdala] = None,
        basal_ganglia: Optional[BasalGanglia] = None,
        cerebellum: Optional[Any] = None,
        motor_cortex: Optional[MotorCortex] = None,
        causal_inference_engine: Optional[Any] = None,
        symbol_grounding: Optional[Any] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        meta_cognitive_snn: Optional[MetaCognitiveSNN] = None,
        world_model: Optional[SpikingWorldModel] = None,
        ethical_guardrail: Optional[EthicalGuardrail] = None,
        reflex_module: Optional[ReflexModule] = None,
        device: Union[str, torch.device] = 'cpu',
        **kwargs: Any
    ):
        super().__init__()
        self.config = config or {}
        self.device = device
        
        # 1. 基礎システムの初期化 (注入がなければデフォルト生成)
        self.workspace = global_workspace if global_workspace else GlobalWorkspace()
        self.motivation_system = motivation_system if motivation_system else IntrinsicMotivationSystem()
        self.astrocyte = astrocyte_network # Optional
        self.guardrail = ethical_guardrail # Optional
        
        # 2. IO
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator

        # 3. 各脳領域
        self.thinking_engine = thinking_engine
        
        # Perception
        if perception_cortex:
            self.perception = perception_cortex
        else:
            # Fallback
            self.perception = PerceptionCortex(num_neurons=784, feature_dim=256)
            
        self.visual_cortex = visual_cortex
        self.amygdala = amygdala if amygdala else Amygdala()
        self.hippocampus = hippocampus if hippocampus else Hippocampus()
        self.cortex = cortex if cortex else Cortex()
        
        # 4. 意思決定系
        self.basal_ganglia = basal_ganglia if basal_ganglia else BasalGanglia(workspace=self.workspace)
        self.prefrontal_cortex = prefrontal_cortex if prefrontal_cortex else PrefrontalCortex(
            workspace=self.workspace, 
            motivation_system=self.motivation_system
        )
        self.pfc = self.prefrontal_cortex # Alias for tests
        
        self.motor = motor_cortex if motor_cortex else MotorCortex()
        self.cerebellum = cerebellum
        
        # 5. 高次認知
        self.reasoning = reasoning_engine
        self.meta_cognitive = meta_cognitive_snn
        self.world_model = world_model
        self.reflex_module = reflex_module
        
        self.cycle_count = 0
        
        # kwargsに含まれる未知のコンポーネントも保持しておく（柔軟性のため）
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def run_cognitive_cycle(self, sensory_input: Union[torch.Tensor, str]) -> Dict[str, Any]:
        """
        1ステップの認知サイクルを実行する。
        """
        self.cycle_count += 1
        # self.deviceがstrの場合もあるため、torch.device化して統一
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        
        # --- 入力の処理 ---
        if isinstance(sensory_input, str):
            # 文字列入力の場合
            # 本来はEncoderを通すべきだが、ここでは簡易的に処理
            # Reasoning Engineがある場合はそちらに任せるなど分岐可能
            if self.reasoning:
                pass
            sensory_tensor = torch.randn(1, 784, device=device) # ダミー
        else:
            sensory_tensor = sensory_input.to(device)

        if sensory_tensor.ndim == 1:
            sensory_tensor = sensory_tensor.unsqueeze(0)

        # 次元の強制整合 (PerceptionCortexの入力数に合わせる)
        if hasattr(self.perception, 'num_neurons'):
            target_n = self.perception.num_neurons
            current_n = sensory_tensor.shape[-1]
            
            if current_n != target_n:
                if current_n < target_n:
                    diff = target_n - current_n
                    padding = torch.zeros(*sensory_tensor.shape[:-1], diff, device=device)
                    sensory_tensor = torch.cat([sensory_tensor, padding], dim=-1)
                else:
                    sensory_tensor = sensory_tensor[..., :target_n]

        # 1. 知覚処理 (Batch, Features)
        perception_result = self.perception.perceive(sensory_tensor)
        perceptual_features = perception_result.get("features")
        
        if perceptual_features is not None:
            perceptual_info = perceptual_features.mean(dim=0) if perceptual_features.ndim > 1 else perceptual_features
        else:
            perceptual_info = torch.zeros(256, device=device)

        # 2. 感情・記憶の評価
        emotional_val = self.amygdala.process(perceptual_info)
        knowledge = self.cortex.retrieve(perceptual_info)
        
        # 3. ワークスペース集約
        self._update_workspace("sensory", perceptual_info)
        self._update_workspace("emotion", emotional_val)
        self._update_workspace("knowledge", knowledge)
        
        # サマリーの取得
        summary = self._get_workspace_summary()
        
        # 4. 行動選択と実行
        # Basal GangliaがWorkspaceのサマリーから行動を選択
        selected_action = self.basal_ganglia.select_action(summary)
        
        # Reflex (反射) の確認
        reflex_triggered = False
        if self.reflex_module:
            reflex_act, conf = self.reflex_module(sensory_tensor)
            if reflex_act is not None:
                selected_action = reflex_act
                reflex_triggered = True

        motor_output = None
        if self.motor:
            motor_output = self.motor.generate_signal(selected_action)

        if hasattr(self.workspace, 'broadcast'):
            self.workspace.broadcast()
            
        # アストロサイトによる代謝制御
        if self.astrocyte:
            self.astrocyte.step()
        
        response = f"Action {selected_action}"
        if reflex_triggered:
            response += " (Reflex Triggered)"

        return {
            "cycle": self.cycle_count,
            "action": {'type': 'reflex' if reflex_triggered else 'voluntary', 'id': selected_action},
            "motor_output": motor_output,
            "knowledge_retrieved": knowledge is not None,
            "broadcasted": True,
            "response": response,
            "status": "success"
        }

    def _update_workspace(self, key: str, value: Any) -> None:
        """ワークスペースへの安全な書き込み"""
        if hasattr(self.workspace, 'add_content'):
            self.workspace.add_content(key, value)
        elif hasattr(self.workspace, 'update'):
            self.workspace.update(key, value)

    def _get_workspace_summary(self) -> List[Dict[str, Any]]:
        """ワークスペースからのサマリー取得"""
        if hasattr(self.workspace, 'get_summary'):
            res = self.workspace.get_summary()
            return cast(List[Dict[str, Any]], res if isinstance(res, list) else [res])
        return []

    def get_brain_status(self) -> Dict[str, Any]:
        """ヘルスチェック用ステータス取得"""
        astro_status = {"status": "unknown", "metrics": {}}
        if self.astrocyte:
            astro_status = {
                "status": "active", 
                "metrics": {
                    "energy_percent": (self.astrocyte.energy / self.astrocyte.max_energy) * 100,
                    "fatigue_index": self.astrocyte.fatigue_toxin
                }
            }
            
        return {
            "cycle": self.cycle_count,
            "status": "active",
            "astrocyte": astro_status
        }

    def get_device(self) -> torch.device:
        if isinstance(self.device, str):
            return torch.device(self.device)
        return self.device