# snn_research/cognitive_architecture/surprise_gated_kernel.py
# 日本語タイトル: Surprise-Gated Brain Kernel v20.2 (Fixes for Sync/Typing)
# 目的: 予測誤差(Surprise)に基づく動的システム切り替えの実装
# 修正: Astrocyte呼び出しからawaitを削除、初期化時の型不整合を修正

import torch
import logging
from typing import Dict, Any, Tuple

from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.safety.ethical_guardrail import EthicalGuardrail
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

logger = logging.getLogger(__name__)

class SurpriseGatedBrain(AsyncArtificialBrain):
    """
    驚き（予測誤差）と感情原子価に基づき、計算リソースを動的に配分する脳カーネル。
    """
    def __init__(self, config: Dict[str, Any]):
        # ローカル変数として初期化し、確実にAstrocyteNetwork型を持つようにする
        astrocyte = AstrocyteNetwork()
        modules: Dict[str, Any] = {} # 初期化時は空、後で設定
        super().__init__(modules=modules, astrocyte=astrocyte)
        
        # BitSpikeMamba の引数をアンパックして渡す
        s1_cfg = config.get("system1_config", {})
        self.system1 = BitSpikeMamba(
            vocab_size=s1_cfg.get("vocab_size", 5000),
            d_model=s1_cfg.get("d_model", 128),
            d_state=s1_cfg.get("d_state", 16),
            d_conv=s1_cfg.get("d_conv", 4),
            expand=s1_cfg.get("expand", 2),
            num_layers=s1_cfg.get("num_layers", 2),
            time_steps=s1_cfg.get("time_steps", 10),
            neuron_config=s1_cfg.get("neuron_config", {"type": "lif"})
        )
        
        # 初期化済みの astrocyte 変数を渡す (self.astrocyte経由だと親クラスの型ヒントの影響を受ける可能性があるため)
        self.system2 = ReasoningEngine(
            generative_model=config.get("reasoning_model"), # type: ignore
            astrocyte=astrocyte 
        )
        
        self.guardrail = EthicalGuardrail(astrocyte=astrocyte)
        self.surprise_threshold = config.get("surprise_threshold", 0.35)
        self.valence_state = 0.0

    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """メタ認知による最適な推論パスの選択と実行"""
        input_data = event.get("data", "")
        
        # 1. 感情的原子価（Valence）の判定
        is_safe, valence_score = await self.guardrail.pre_check(str(input_data))
        self.valence_state = 0.8 * self.valence_state + 0.2 * valence_score

        if not is_safe:
            # Astrocyteメソッドは同期呼び出しに変更 (await削除)
            self.astrocyte.consume_energy("REFLEX_REJECTION")
            return {"type": "ACTION", "data": self.guardrail.generate_gentle_refusal("safety violation")}

        # 2. System 1 による予測誤差の算出
        s1_output, prediction_error = await self._run_system1_logic(input_data)
        
        # 3. メタ認知ゲート: Surprise 判定
        if prediction_error > self.surprise_threshold:
            logger.info(f"🤔 Surprise detected ({prediction_error:.2f}). Activating System 2...")
            
            # Astrocyteメソッドは同期呼び出し (await削除)
            if self.astrocyte.request_compute_boost():
                # input_data は Tensor である必要があるが、ここでは簡易化
                dummy_ids = torch.randint(0, 10, (1, 5)) 
                res = self.system2.think_and_solve(dummy_ids)
                final_output = f"System 2 Analysis: {res.get('strategy')}"
                self.astrocyte.log_fatigue(0.15)
            else:
                final_output = f"Fallback (Energy Low): {s1_output}"
        else:
            final_output = s1_output
            self.astrocyte.consume_energy("SYSTEM1_IDLE")

        return {
            "type": "THOUGHT_RESPONSE",
            "data": final_output,
            "meta": {
                "surprise": prediction_error,
                "valence": self.valence_state,
                "mode": "System 2" if prediction_error > self.surprise_threshold else "System 1"
            }
        }

    async def _run_system1_logic(self, data: Any) -> Tuple[str, float]:
        """BitSpikeMamba推論と擬似予測誤差の生成"""
        # 本来は Spike Activity からエントロピーを計算
        return "System 1 Intuition", torch.rand(1).item()