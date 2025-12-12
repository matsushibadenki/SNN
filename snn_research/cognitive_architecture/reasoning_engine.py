# ファイルパス: snn_research/cognitive_architecture/reasoning_engine.py
# 日本語タイトル: Reasoning Engine with GRPO & Verifier (System 2 Thinking Loop) [Fixed]
# 目的・内容:
#   ROADMAP v16 (Phase 17.5) "Thinking & Verifier" の実装。
#   修正: mypyエラー "Tensor not callable" を解消するために cast を導入。
#   推論結果の自己検証(Verifier)と、GRPO的な思考パス選択を行う。

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, cast
import logging

from snn_research.models.transformer.sformer import SFormer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

logger = logging.getLogger(__name__)

class VerifierNetwork(nn.Module):
    """
    思考(Thought)と結論(Answer)のペアを入力とし、その妥当性スコア(0.0-1.0)を出力する軽量SNN。
    """
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.compressor = nn.Linear(d_model, hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (Batch, SeqLen, d_model) -> (Batch, 1)
        pooled = torch.mean(hidden_states, dim=1)
        compressed = self.compressor(pooled)
        score = self.value_head(compressed)
        return score


class ReasoningEngine:
    """
    熟慮（Thinking）と検証（Verifier）のループを制御するエンジン。
    """
    def __init__(
        self,
        generative_model: SFormer,
        astrocyte: AstrocyteNetwork,
        verifier_model: Optional[VerifierNetwork] = None,
        d_model: int = 256,
        num_thinking_paths: int = 4, 
        max_thinking_steps: int = 32,
        device: str = 'cpu'
    ):
        self.model = generative_model
        self.astrocyte = astrocyte
        self.device = device
        
        if verifier_model is None:
            self.verifier = VerifierNetwork(d_model=d_model).to(device)
        else:
            self.verifier = verifier_model.to(device)
            
        self.num_thinking_paths = num_thinking_paths
        self.max_thinking_steps = max_thinking_steps
        
        logger.info(f"🧠 ReasoningEngine initialized (Paths: {num_thinking_paths}, Device: {device})")

    def think_and_solve(
        self, 
        input_ids: torch.Tensor, 
        task_type: str = "general",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Thinking Loop: 複数の思考パスを生成し、Verifierで評価して選択する。
        """
        B = input_ids.shape[0]
        if B != 1:
            raise ValueError("ReasoningEngine currently supports batch_size=1 only.")

        # 1. エネルギーリソース確認
        estimated_cost = self.num_thinking_paths * self.max_thinking_steps * 0.1
        module_name = "cortex" if task_type == "general" else "prefrontal_cortex"
        
        if not self.astrocyte.request_resource(module_name, estimated_cost):
            logger.info("⚡ Low Energy: Fallback to System 1.")
            return self._system1_inference(input_ids)

        # 2. System 2: Thinking Loop
        logger.info(f"🤔 Thinking... Generating {self.num_thinking_paths} paths.")
        candidates = []
        batch_input = input_ids.repeat(self.num_thinking_paths, 1) # (N, L)
        
        self.model.eval()
        self.verifier.eval()
        
        with torch.no_grad():
            # 2.1 思考パスの並列生成
            # mypyエラー回避: cast(Any, self.model).generate(...)
            # SFormerにgenerateメソッドを追加したため、実行時は安全
            gen_model = cast(Any, self.model)
            
            if hasattr(gen_model, 'generate'):
                generated_outputs = gen_model.generate(
                    batch_input, 
                    max_length=self.max_thinking_steps + input_ids.shape[1],
                    temperature=temperature,
                    do_sample=True
                ) # (N, TotalLen)
            else:
                return self._system1_inference(input_ids)

            # 2.2 検証 (Verifier)
            # LogitsやHidden Statesを取得するためにForwardを通す
            # SFormer.forward -> (logits, spikes, mem)
            # ここではLogitsをEmbeddingして疑似Hiddenとするか、
            # 将来的にはSFormerがHiddenを返すようにする。暫定でEmbeddingレイヤを通すかLogits使用。
            # 簡易的にEmbedding層を再利用して近似
            
            # (N, TotalLen) -> (N, TotalLen, D)
            hidden_approx = self.model.embedding(generated_outputs) 
            scores = self.verifier(hidden_approx)
            scores_list = scores.view(-1).tolist()
            
            for i in range(self.num_thinking_paths):
                candidates.append({
                    "output_ids": generated_outputs[i],
                    "score": scores_list[i],
                    "path_id": i
                })

        # 3. Selection
        best_candidate = max(candidates, key=lambda x: x["score"])
        
        return {
            "final_output": best_candidate["output_ids"].unsqueeze(0),
            "thought_trace": [f"Path {c['path_id']}: {c['score']:.3f}" for c in candidates],
            "verifier_score": best_candidate["score"],
            "strategy": "system2_verified"
        }

    def _system1_inference(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """System 1: 即時推論 (Greedy)"""
        self.model.eval()
        gen_model = cast(Any, self.model)
        
        with torch.no_grad():
            if hasattr(gen_model, 'generate'):
                output = gen_model.generate(
                    input_ids, 
                    max_length=input_ids.shape[1] + 16,
                    do_sample=False
                )
            else:
                logits, _, _ = self.model(input_ids)
                output = torch.argmax(logits, dim=-1)
                
        return {
            "final_output": output,
            "strategy": "system1_fast"
        }
