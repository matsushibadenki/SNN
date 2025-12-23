# /snn_research/agent/active_inference_agent.py
# 日本語タイトル: 能動的推論エージェント (Active Inference Agent) v3.0 EFE-Enhanced
# 目的・内容: 
#   自由エネルギー原理（FEP）に基づき、知覚、意思決定、学習を統合する自律型エージェント。
#   修正: 期待自由エネルギー(EFE)に基づく動的な探索・利用のバランス制御を追加。

import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import logging

# 既存の依存コンポーネント
from snn_research.core.snn_core import SNNCore
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.utils.brain_debugger import BrainDebugger

logger = logging.getLogger(__name__)

class ActiveInferenceAgent:
    """
    能動的推論（Active Inference）を司る自律エージェント。
    
    Update v3.0:
    - Expected Free Energy (EFE): 
      G = (Information Gain) + (Goal Realization)
      不確実性が高い場合は「知識獲得」を優先し、低い場合は「ゴール達成」を優先する。
    """

    def __init__(
        self,
        brain_kernel: Any,
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
        self.epistemic_weight = config.get("epistemic_weight", 1.0) # 好奇心の強さ
        self.is_running = False

    def _calculate_expected_free_energy(self, confidence: float, surprise: float) -> float:
        """
        簡易的な期待自由エネルギー(EFE)の計算。
        G ≈ - (Epistemic Value) - (Extrinsic Value)
        ここではコスト（最小化すべき値）として計算。
        """
        # 認識的価値（情報利得）: 不確実性が高いほど、それを解消する行動の価値が高い
        # 自信が低い(confidence小) -> 情報利得の余地が大きい
        epistemic_value = (1.0 - confidence) * self.epistemic_weight
        
        # 外的価値（実用的価値）: 驚きが小さい（予測通り）状態を好む
        # ここでは簡易的に「驚きの小ささ」を価値とする
        extrinsic_value = -surprise 
        
        # EFE (負の値が大きいほど良い行動、ここではコストとして正の値に変換して扱うことも可能だが、
        # 単純にトリガー判定に使う)
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
        # エントロピーを驚きの近似として使用
        efe_score = self._calculate_expected_free_energy(confidence, entropy)
        
        # System 2 トリガー条件の高度化:
        # 単なる信頼度だけでなく、EFEに基づいて「深く考える価値があるか」を判断
        # EFEが低い（負に大きい）＝ 情報獲得の価値が高い or リスクが高い
        trigger_system2 = meta_stats.get("trigger_system2", False)
        
        # 不確実性が高く、かつ情報獲得の価値がある場合にSystem 2を起動
        if efe_score < -0.5: 
            trigger_system2 = True

        logger.debug(f"🧠 Meta: Conf={confidence:.2f}, EFE={efe_score:.2f} -> Sys2={trigger_system2}")

        # 3. 推論モードの実行
        if not trigger_system2 and confidence > self.confidence_threshold:
            # --- System 1: Intuition ---
            output = system1_logits
            inference_type = "System_1_Intuition"
            
            firing_rates = self.snn_core.get_firing_rates()
            self.astrocyte.monitor_neural_activity(firing_rates)
        else:
            # --- System 2: Reasoning ---
            cost = 20.0
            if self.astrocyte.request_resource("reasoning_engine", cost):
                logger.info("🤔 High EFE/Low Conf. Activating System 2...")
                
                # 推論エンジンにEFE情報を渡すことも可能
                output_dict = self.reasoning_engine.process(sensory_input)
                output = output_dict.get("final_output", system1_logits)
                inference_type = "System_2_Reasoning"
            else:
                logger.warning("⚡ Low Energy: System 2 denied. Fallback.")
                output = system1_logits
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

        # 5. 能動的探索 (Active Exploration)
        # EFEに基づいて「知識ギャップ」イベントを発行
        if trigger_system2 and self.epistemic_weight > 0:
            await self.kernel.publish(BrainEvent(
                event_type="KNOWLEDGE_GAP_DETECTED",
                source="active_inference_agent",
                payload={
                    "input": sensory_input, 
                    "confidence": confidence,
                    "efe": efe_score
                }
            ))

        return action_results

    async def run_autonomous_loop(self):
        """自律稼働ループ"""
        self.is_running = True
        logger.info("🤖 Active Inference Agent Loop Started.")
        while self.is_running:
            try:
                if self.kernel.state == "SLEEP":
                    await asyncio.sleep(5.0)
                    continue
                await asyncio.sleep(0.1) 
            except Exception as e:
                logger.error(f"Error in Agent Loop: {e}")
                await asyncio.sleep(1.0)

    def stop(self):
        self.is_running = False

# ダミーイベントクラス（インポートエラー回避用）
class BrainEvent:
    def __init__(self, event_type, source, payload):
        self.event_type = event_type
        self.source = source
        self.payload = payload
