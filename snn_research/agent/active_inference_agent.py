# /snn_research/agent/active_inference_agent.py
# 日本語タイトル: 能動的推論エージェント (Active Inference Agent) v3.1 Formalized
# 目的・内容: 
#   ダミーコード(BrainEvent等)を正式実装し、型安全性と拡張性を向上。
#   カーネルとのインターフェースをプロトコルとして定義。

import asyncio
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Protocol, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import time

# 既存の依存コンポーネント
from snn_research.core.snn_core import SNNCore
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.utils.brain_debugger import BrainDebugger

logger = logging.getLogger(__name__)

# --- [正式実装] イベントシステム定義 ---

class BrainEventType(Enum):
    """脳内イベントの種類"""
    SENSORY_INPUT = auto()
    MOTOR_COMMAND = auto()
    KNOWLEDGE_GAP_DETECTED = auto() # 知識不足検知
    PREDICTION_ERROR = auto()       # 予測誤差
    SYSTEM2_ACTIVATION = auto()     # 熟慮モード起動
    SLEEP_REQUEST = auto()          # 睡眠要求

@dataclass
class BrainEvent:
    """
    脳内イベントデータの正式定義。
    """
    event_type: Union[BrainEventType, str]
    source: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 1 # 1: Low, 5: High

# --- [正式実装] カーネルインターフェース定義 ---

@runtime_checkable
class AbstractBrainKernel(Protocol):
    """
    BrainKernelが満たすべきインターフェース定義。
    依存関係逆転の原則に基づき、Agentは具象クラスではなくこのプロトコルに依存する。
    """
    state: str
    
    async def publish(self, event: BrainEvent) -> None:
        ...
        
    async def subscribe(self, event_type: BrainEventType, callback: Any) -> None:
        ...

# --- Agent Implementation ---

class ActiveInferenceAgent:
    """
    能動的推論（Active Inference）を司る自律エージェント。
    """

    def __init__(
        self,
        brain_kernel: AbstractBrainKernel, # 型ヒントをプロトコルに変更
        config: Dict[str, Any],
        snn_core: SNNCore,
        reasoning_engine: ReasoningEngine,
        astrocyte: AstrocyteNetwork
    ):
        self.kernel = brain_kernel
        self.config = config
        self.debugger = BrainDebugger()
        
        self.snn_core = snn_core
        self.reasoning_engine = reasoning_engine
        self.astrocyte = astrocyte
        
        self.meta_cognition = MetaCognitiveSNN(
            d_model=config.get("d_model", 128),
            uncertainty_threshold=config.get("uncertainty_threshold", 0.6)
        )
        
        self.confidence_threshold = config.get("confidence_threshold", 0.85)
        self.epistemic_weight = config.get("epistemic_weight", 1.0)
        self.is_running = False

    def _calculate_expected_free_energy(self, confidence: float, surprise: float) -> float:
        """期待自由エネルギー(EFE)の計算"""
        # EFE G ≈ - (Information Gain) - (Extrinsic Value)
        epistemic_value = (1.0 - confidence) * self.epistemic_weight
        extrinsic_value = -surprise 
        G = -epistemic_value - extrinsic_value
        return G

    async def step(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        """
        1タイムステップの推論・行動ループ。
        """
        # 1. 知覚とメタ認知 (System 1)
        system1_logits = self.snn_core.forward(sensory_input)
        if isinstance(system1_logits, tuple):
            system1_logits = system1_logits[0]

        meta_stats = self.meta_cognition.monitor_system1_output(system1_logits)
        confidence = meta_stats.get("confidence", 0.0)
        entropy = meta_stats.get("entropy", 0.0)
        
        # 2. 期待自由エネルギーによる評価
        efe_score = self._calculate_expected_free_energy(confidence, entropy)
        
        # System 2 トリガー判定
        trigger_system2 = meta_stats.get("trigger_system2", False)
        if efe_score < -0.5: # 情報獲得価値が高い場合もトリガー
            trigger_system2 = True

        logger.debug(f"🧠 Meta: Conf={confidence:.2f}, EFE={efe_score:.2f} -> Sys2={trigger_system2}")

        # 3. 推論モードの実行
        inference_type = "System_1_Intuition"
        output = system1_logits

        if not trigger_system2 and confidence > self.confidence_threshold:
            # System 1: Intuition
            firing_rates = self.snn_core.get_firing_rates()
            self.astrocyte.monitor_neural_activity(firing_rates)
        else:
            # System 2: Reasoning
            cost = 20.0
            if self.astrocyte.request_resource("reasoning_engine", cost):
                logger.info("🤔 High EFE/Low Conf. Activating System 2...")
                
                # イベント発行: System 2 起動
                await self.kernel.publish(BrainEvent(
                    event_type=BrainEventType.SYSTEM2_ACTIVATION,
                    source="active_inference_agent",
                    payload={"reason": "low_confidence_or_high_efe", "efe": efe_score},
                    priority=3
                ))

                output_dict = self.reasoning_engine.process(sensory_input)
                output = output_dict.get("final_output", system1_logits)
                inference_type = "System_2_Reasoning"
            else:
                logger.warning("⚡ Low Energy: System 2 denied. Fallback.")
                inference_type = "System_1_Fallback"

        # 4. 状態報告
        diagnosis = self.astrocyte.get_diagnosis_report()
        self.debugger.explain_thought_process(
            input_text="Sensory Stream", 
            output_text=str(inference_type),
            astrocyte_status=diagnosis
        )

        action_results = {
            "inference_type": inference_type,
            "confidence": confidence,
            "efe_score": efe_score,
            "output": output,
            "astrocyte_report": diagnosis
        }

        # 5. 能動的探索 (Knowledge Gap Event)
        # EFEに基づいて「知識ギャップ」イベントを発行
        if trigger_system2 and self.epistemic_weight > 0:
            await self.kernel.publish(BrainEvent(
                event_type=BrainEventType.KNOWLEDGE_GAP_DETECTED,
                source="active_inference_agent",
                payload={
                    "input_sample": sensory_input.detach().cpu().numpy().tolist() if hasattr(sensory_input, 'detach') else str(sensory_input), 
                    "confidence": confidence,
                    "efe": efe_score
                },
                priority=4
            ))

        return action_results

    async def run_autonomous_loop(self):
        """自律稼働ループ"""
        self.is_running = True
        logger.info("🤖 Active Inference Agent Loop Started.")
        while self.is_running:
            try:
                # カーネルの状態確認（プロトコル経由）
                if getattr(self.kernel, 'state', '') == "SLEEP":
                    await asyncio.sleep(5.0)
                    continue
                await asyncio.sleep(0.1) 
            except Exception as e:
                logger.error(f"Error in Agent Loop: {e}")
                await asyncio.sleep(1.0)

    def stop(self):
        self.is_running = False
