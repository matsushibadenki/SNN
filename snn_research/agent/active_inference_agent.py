# /snn_research/agent/active_inference_agent.py
# 能動的推論エージェント (Active Inference Agent)
# 目的: 自由エネルギー原理に基づき、知覚（予測誤差の最小化）と行動（能動的探索）を統合する。
#      メタ認知によって、System 1 (直感) と System 2 (熟慮) を動的に切り替える。

import asyncio
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from snn_research.core.snn_core import SNNCore
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.utils.brain_debugger import BrainDebugger

class ActiveInferenceAgent:
    """
    能動的推論（Active Inference）を司るエージェントクラス。
    
    ロジック:
    1. 外部刺激をスパイクとして受容。
    2. MetaCognitiveSNNが現在の入力に対する「驚き（Surprise/Entropy）」を算出。
    3. 驚きが閾値以下なら System 1 (BitSpikeMamba等) で即時応答。
    4. 驚きが高い、あるいは自信がない場合は System 2 (ReasoningEngine) を起動し、
       RAGやシミュレーションを用いて「深く考える」。
    5. Astrocyte Networkを通じて消費エネルギーを監視し、過負荷時は思考を抑制する。
    """

    def __init__(
        self,
        brain_kernel: Any,
        config: Dict[str, Any]
    ):
        self.kernel = brain_kernel
        self.config = config
        self.debugger = BrainDebugger()
        
        # コンポーネントの初期化
        self.snn_core = SNNCore(config.get("snn_config"))
        self.meta_cognition = MetaCognitiveSNN()
        self.reasoning_engine = ReasoningEngine()
        self.astrocyte = AstrocyteNetwork()
        
        self.confidence_threshold = config.get("confidence_threshold", 0.85)
        self.is_running = False

    async def step(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        """
        1タイムステップの推論・行動ループ
        """
        # 0. エネルギー状態の確認 (Astrocyteによる制約)
        energy_status = await self.astrocyte.check_homeostasis()
        if energy_status["fatigue"] > 0.9:
            return {"action": "sleep", "reason": "High metabolic fatigue"}

        # 1. 知覚とメタ認知 (Surpriseの計測)
        # 予測誤差（自由エネルギー）の近似としてエントロピーを使用
        meta_stats = await self.meta_cognition.evaluate_uncertainty(sensory_input)
        surprise = meta_stats.get("surprise", 1.0)
        confidence = 1.0 - surprise

        self.debugger.log(f"Input Surprise: {surprise:.4f} | Confidence: {confidence:.4f}")

        # 2. 推論モードの動的選択 (Dynamic Compute)
        if confidence > self.confidence_threshold:
            # --- System 1: 高速・省エネ推論 (直感) ---
            # BitSpikeMamba等の軽量モデルを使用
            output = await self.snn_core.forward_async(sensory_input, mode="fast")
            inference_type = "System_1_Intuition"
        else:
            # --- System 2: 深層推論 (熟慮) ---
            # ReasoningEngineによる多段階推論と検証
            self.debugger.log("Confidence low. Activating System 2 Reasoning Engine...")
            
            # 思考ブーストによるエネルギー消費の増加をAstrocyteに通知
            await self.astrocyte.allocate_resources(priority="high", task="deep_reasoning")
            
            # RAG検索や内部シミュレーション（World Model）を伴う推論
            output = await self.reasoning_engine.reason_with_verification(
                context=sensory_input,
                steps=self.config.get("max_reasoning_steps", 5)
            )
            inference_type = "System_2_Reasoning"

        # 3. 行動の決定と出力
        action_results = {
            "inference_type": inference_type,
            "confidence": confidence,
            "output": output,
            "metabolic_cost": await self.astrocyte.get_current_consumption()
        }

        # 4. 学習プロセス (Predictive Codingに基づく局所更新)
        # 本来は非同期でSleepConsolidatorに回すが、ここでは即時学習のトリガーのみ
        if surprise > 0.5:
            await self.kernel.trigger_event("curiosity_learning", data=action_results)

        return action_results

    async def run_autonomous_loop(self):
        """自律稼働ループ"""
        self.is_running = True
        while self.is_running:
            # センサーからの入力をシミュレートまたは取得
            sensory_data = await self.kernel.get_next_sensory_event()
            if sensory_data:
                result = await self.step(sensory_data)
                await self.kernel.publish_action(result)
            
            await asyncio.sleep(0.01) # 10ms レイテンシ維持目標

    def stop(self):
        self.is_running = False

# ロジックの正当性確認:
# - Objective.md ⑮(動的リソース配分)を、Surpriseベースの分岐で実装
# - Roadmap.md の Async First 原則に基づき、すべて asyncio で非同期化
# - Biomimetic...md の「予測誤差のみを上位へ」というPCの考えをメタ認知のトリガーに採用
