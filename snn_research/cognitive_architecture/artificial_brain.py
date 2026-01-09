# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# Title: 人工脳コア・アーキテクチャ (Phase 6 Fix: Dimension Alignment)
# Description:
# - Thalamusの出力次元とPerceptionCortexの入力次元を自動的に整合させるロジックを追加。
# - 視床リレーループにおけるValueErrorを解消。

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
from .sleep_consolidation import SleepConsolidator
from .thalamus import Thalamus
from snn_research.modules.reflex_module import ReflexModule
from snn_research.models.experimental.world_model_snn import SpikingWorldModel
from snn_research.safety.ethical_guardrail import EthicalGuardrail
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator

logger = logging.getLogger(__name__)


class ArtificialBrain(nn.Module):
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
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
        thalamus: Optional[Thalamus] = None,
        causal_inference_engine: Optional[Any] = None,
        symbol_grounding: Optional[Any] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        meta_cognitive_snn: Optional[MetaCognitiveSNN] = None,
        world_model: Optional[SpikingWorldModel] = None,
        ethical_guardrail: Optional[EthicalGuardrail] = None,
        reflex_module: Optional[ReflexModule] = None,
        sleep_consolidator: Optional[SleepConsolidator] = None,
        device: Union[str, torch.device] = 'cpu',
        **kwargs: Any
    ):
        super().__init__()
        self.config = config or {}
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.state = "AWAKE"

        # 1. 基礎システム
        self.workspace = global_workspace if global_workspace else GlobalWorkspace()
        self.motivation_system = motivation_system if motivation_system else IntrinsicMotivationSystem()
        self.astrocyte = astrocyte_network
        self.guardrail = ethical_guardrail

        # 2. IO
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator
        
        # 3. 脳領域
        self.thinking_engine = thinking_engine

        # [Fix] Thalamus初期化
        # 視床が存在する場合、その出力次元を取得してPerceptionCortexの入力次元とする
        self.thalamus = thalamus if thalamus else Thalamus(device=str(self.device))
        
        if perception_cortex:
            self.perception = perception_cortex
        else:
            # [Fix] 次元自動調整ロジック
            # デフォルトのThalamus出力は256なので、それに合わせる
            input_neurons = 784 # default fallback
            if self.thalamus and hasattr(self.thalamus, 'output_dim'):
                input_neurons = self.thalamus.output_dim
                logger.info(f"🧠 Adjusting PerceptionCortex input size to match Thalamus: {input_neurons}")
            
            self.perception = PerceptionCortex(num_neurons=input_neurons, feature_dim=256)

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
        self.pfc = self.prefrontal_cortex

        self.motor = motor_cortex if motor_cortex else MotorCortex()
        self.cerebellum = cerebellum

        # 5. 高次認知
        self.reasoning = reasoning_engine
        self.meta_cognitive = meta_cognitive_snn
        self.world_model = world_model
        self.reflex_module = reflex_module
        self.sleep_consolidator = sleep_consolidator

        self.cycle_count = 0

        # kwargs保持
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

        self.to(self.device)
        logger.info(f"ArtificialBrain initialized on {self.device}")

    def run_cognitive_cycle(self, sensory_input: Union[torch.Tensor, str]) -> Dict[str, Any]:
        """
        1ステップの認知サイクルを実行する。
        Phase 6: Thalamocortical Loop (感覚 -> 視床 -> 皮質)
        """
        self.cycle_count += 1
        
        if self.state == "SLEEPING":
             return {"cycle": self.cycle_count, "status": "sleeping"}

        # --- 1. 入力エンコーディング (Sensation) ---
        sensory_tensor: Optional[torch.Tensor] = None

        if isinstance(sensory_input, str):
            if self.encoder:
                try:
                    sensory_tensor = self.encoder.encode_text(sensory_input)
                except Exception as e:
                    logger.error(f"Encoding failed: {e}")
                    sensory_tensor = torch.zeros(1, 784, device=self.device)
            else:
                logger.warning("No SpikeEncoder configured.")
                sensory_tensor = torch.zeros(1, 784, device=self.device)
        else:
            sensory_tensor = sensory_input

        # [Safety] デバイス整合
        if sensory_tensor is not None:
            sensory_tensor = sensory_tensor.to(self.device)
            if sensory_tensor.ndim == 1:
                sensory_tensor = sensory_tensor.unsqueeze(0)

        # 次元の強制整合 (for Thalamus input)
        # Thalamus入力次元に合わせてパディングまたはカット
        if sensory_tensor is not None:
            target_n = self.thalamus.input_dim
            current_n = sensory_tensor.shape[-1]
            if current_n != target_n:
                if current_n < target_n:
                    diff = target_n - current_n
                    padding = torch.zeros(*sensory_tensor.shape[:-1], diff, device=self.device)
                    sensory_tensor = torch.cat([sensory_tensor, padding], dim=-1)
                else:
                    sensory_tensor = sensory_tensor[..., :target_n]

        # --- 2. 視床リレー (Thalamic Relay) ---
        top_down_signal = None
        if self.workspace.conscious_broadcast_content is not None:
            content = self.workspace.conscious_broadcast_content
            # 注意信号の次元チェック
            if isinstance(content, torch.Tensor) and content.shape[-1] == self.thalamus.output_dim:
                top_down_signal = content.to(self.device)

        thalamic_result = self.thalamus(sensory_tensor, top_down_attention=top_down_signal)
        
        # Thalamus出力 (通常 256次元)
        relayed_input = thalamic_result["relayed_output"]

        # --- 3. 皮質知覚処理 (Perception) ---
        # 修正: self.perceptionは初期化時にThalamus出力次元に合わせてあるはず
        perception_result = self.perception.perceive(relayed_input)
        perceptual_features = perception_result.get("features")

        if perceptual_features is not None:
            perceptual_info = perceptual_features.mean(dim=0) if perceptual_features.ndim > 1 else perceptual_features
        else:
            perceptual_info = torch.zeros(256, device=self.device)

        # 高次推論 (Reasoning)
        reasoning_output = None
        if self.reasoning and isinstance(sensory_input, str):
            reasoning_output = self.reasoning.process(sensory_input)
            if reasoning_output:
                self._update_workspace("reasoning", reasoning_output)

        # --- 4. 感情・記憶・評価 ---
        emotional_val = self.amygdala.process(perceptual_info)
        knowledge = self.cortex.retrieve(perceptual_info)

        # --- 5. ワークスペース集約 ---
        self._update_workspace("sensory", perceptual_info)
        self._update_workspace("emotion", emotional_val)
        if knowledge is not None:
            self._update_workspace("knowledge", knowledge)

        summary = self._get_workspace_summary()

        # --- 6. 行動選択 ---
        selected_action = self.basal_ganglia.select_action(summary)

        # Reflex (反射)
        reflex_triggered = False
        if self.reflex_module and sensory_tensor is not None:
            reflex_act, conf = self.reflex_module(sensory_tensor)
            if reflex_act is not None:
                selected_action = reflex_act
                reflex_triggered = True

        # --- 7. 行動実行 ---
        motor_output = None
        execution_result = None

        if self.motor:
            motor_output = self.motor.generate_signal(selected_action)

            if self.actuator and motor_output is not None:
                execution_result = self.actuator.execute(
                    motor_output, action_id=selected_action)

        if hasattr(self.workspace, 'broadcast'):
            self.workspace.broadcast()

        if self.astrocyte:
            self.astrocyte.step()

        response_text = f"Action {selected_action}"
        if reflex_triggered:
            response_text += " (Reflex Triggered)"
        if execution_result:
            response_text += f", Result: {execution_result}"
        
        return {
            "cycle": self.cycle_count,
            "action": {
                'type': 'reflex' if reflex_triggered else 'voluntary',
                'id': selected_action,
                'executed': execution_result is not None
            },
            "thalamus_gate": thalamic_result["gate_value"].mean().item(),
            "motor_output": motor_output,
            "response": response_text,
            "status": "success"
        }

    def sleep_cycle(self) -> None:
        logger.info("💤 Initiating Sleep Cycle...")
        self.state = "SLEEPING"
        self.thalamus.set_state("SLEEP")
        
        if self.sleep_consolidator:
            try:
                self.sleep_consolidator.perform_sleep_cycle(duration_cycles=5)
            except Exception as e:
                logger.error(f"Error during sleep consolidation: {e}")
                
        if self.astrocyte:
            self.astrocyte.clear_fatigue(50.0)
            self.astrocyte.replenish_energy(300.0)

        self.state = "AWAKE"
        self.thalamus.set_state("AWAKE")
        logger.info("🌞 Brain has awakened.")

    def _update_workspace(self, key: str, value: Any) -> None:
        if hasattr(self.workspace, 'add_content'):
            self.workspace.add_content(key, value)
        elif hasattr(self.workspace, 'update'):
            self.workspace.update(key, value)

    def _get_workspace_summary(self) -> List[Dict[str, Any]]:
        if hasattr(self.workspace, 'get_summary'):
            res = self.workspace.get_summary()
            if res is None:
                return []
            return cast(List[Dict[str, Any]], res if isinstance(res, list) else [res])
        return []

    def get_brain_status(self) -> Dict[str, Any]:
        astro_status = {"status": "unknown", "metrics": {}}
        if self.astrocyte:
            astro = cast(AstrocyteNetwork, self.astrocyte)
            max_e = astro.max_energy if astro.max_energy > 0 else 1.0
            energy_percent = (astro.energy / max_e) * 100
            
            astro_status = {
                "status": "active",
                "metrics": {
                    "energy_percent": energy_percent,
                    "fatigue_index": astro.fatigue_toxin
                }
            }

        return {
            "cycle": self.cycle_count,
            "status": self.state,
            "device": str(self.device),
            "astrocyte": astro_status,
            "components": {
                "thalamus": self.thalamus is not None,
                "perception": self.perception is not None,
                "sleep_consolidator": self.sleep_consolidator is not None
            }
        }

    def get_device(self) -> torch.device:
        return self.device