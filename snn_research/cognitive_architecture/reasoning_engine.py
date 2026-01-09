# ファイルパス: snn_research/cognitive_architecture/reasoning_engine.py
# 日本語タイトル: Reasoning Engine v2.7 - Type Fixes
# 目的: Mypyエラー ("Tensor" not callable) の修正と型安全性の向上。

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, cast, Tuple
import logging
import re
import io
import contextlib
import multiprocessing
import os
from transformers import PreTrainedTokenizerBase

# 依存関係
from snn_research.models.transformer.sformer import SFormer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.rag_snn import RAGSystem

logger = logging.getLogger(__name__)

class CodeSandbox:
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.forbidden_modules = [
            "os", "sys", "subprocess", "shutil", "netrc", "requests", "urllib",
            "socket", "pathlib", "input", "open"
        ]

    def _execute_code(self, code: str, queue: multiprocessing.Queue):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        buffer = io.StringIO()
        for mod in self.forbidden_modules:
            if f"import {mod}" in code or f"from {mod}" in code:
                queue.put((False, f"Security Error: Usage of '{mod}' is forbidden."))
                return
        try:
            with contextlib.redirect_stdout(buffer):
                local_scope: Dict[str, Any] = {}
                exec(code, {}, local_scope)
            output = buffer.getvalue()
            queue.put((True, output if output else "Executed successfully (no output)."))
        except Exception as e:
            queue.put((False, f"Runtime Error: {str(e)}"))

    def run(self, code: str) -> Tuple[bool, str]:
        queue: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=self._execute_code, args=(code, queue))
        p.start()
        p.join(self.timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            return False, "Timeout Error: Code execution took too long."
        if not queue.empty():
            return queue.get()
        return False, "Unknown Error: No result returned."

class VerifierNetwork(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.mean(dim=1))

class ReasoningEngine:
    def __init__(
        self,
        generative_model: SFormer,
        astrocyte: AstrocyteNetwork,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        rag_system: Optional[RAGSystem] = None,
        verifier_model: Optional[VerifierNetwork] = None,
        d_model: int = 256,
        num_thinking_paths: int = 3,
        max_thinking_steps: int = 128,
        enable_code_verification: bool = True,
        enable_rag_verification: bool = True,
        sandbox_timeout: float = 10.0,
        max_retries: int = 2,
        device: str = 'cpu'
    ):
        self.model = generative_model
        # 明示的に型を指定
        self.astrocyte: AstrocyteNetwork = astrocyte
        self.tokenizer = tokenizer
        self.rag_system = rag_system
        self.device = device
        self.enable_code_verification = enable_code_verification
        self.enable_rag_verification = enable_rag_verification
        self.max_retries = max_retries

        if verifier_model is None:
            self.verifier = VerifierNetwork(d_model=d_model).to(device)
        else:
            self.verifier = verifier_model.to(device)

        self.num_thinking_paths = num_thinking_paths
        self.max_thinking_steps = max_thinking_steps
        self.sandbox = CodeSandbox(timeout=sandbox_timeout)

        logger.info("🧠 ReasoningEngine v2.7 initialized.")

    def process(self, input_data: Any) -> Dict[str, Any]:
        input_ids = None
        current_tokenizer = self.tokenizer

        try:
            if isinstance(input_data, str):
                if self.tokenizer is None:
                    return {"error": "Tokenizer required", "strategy": "none"}
                encoded = self.tokenizer(input_data, return_tensors="pt")
                input_ids = encoded.input_ids.to(self.device)
            elif isinstance(input_data, dict):
                if "input_ids" in input_data:
                    input_ids = input_data["input_ids"].to(self.device)
                elif "text" in input_data and self.tokenizer:
                    encoded = self.tokenizer(input_data["text"], return_tensors="pt")
                    input_ids = encoded.input_ids.to(self.device)
                if "tokenizer" in input_data:
                    current_tokenizer = input_data["tokenizer"]
            elif isinstance(input_data, torch.Tensor):
                input_ids = input_data.to(self.device)

            if input_ids is None:
                return {"error": "Could not extract input_ids", "strategy": "none"}

            return self.think_and_solve(input_ids=input_ids, tokenizer=current_tokenizer)
        except Exception as e:
            logger.error(f"ReasoningEngine Process Error: {e}", exc_info=True)
            return {"error": str(e), "strategy": "error_recovery"}

    def _extract_query(self, text: str) -> Optional[str]:
        match = re.search(r"<query>(.*?)</query>", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_code_blocks(self, text: str) -> List[str]:
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches if m.strip()]

    def think_and_solve(
        self,
        input_ids: torch.Tensor,
        task_type: str = "general",
        temperature: float = 0.7,
        tokenizer: Any = None
    ) -> Dict[str, Any]:

        input_text = ""
        if tokenizer:
            input_text = tokenizer.decode(input_ids[0]).lower()

        force_system2 = any(w in input_text for w in ["calculate", "code", "python", "計算", "書いて", "足し算"])

        estimated_cost = float(self.num_thinking_paths * self.max_thinking_steps * 0.2)
        
        # [Fix] 明示的なメソッド呼び出しと型チェック
        has_energy = self.astrocyte.request_resource("prefrontal_cortex", estimated_cost)

        if not force_system2 and not has_energy:
            return self._system1_inference(input_ids)

        logger.info(f"🤔 System 2 Activated (Force: {force_system2})")

        candidates = []
        gen_model = cast(Any, self.model)

        for path_idx in range(self.num_thinking_paths):
            current_input_ids = input_ids.clone()
            path_trace: List[str] = []

            for attempt in range(self.max_retries + 1):
                generated_ids, rag_log, current_text = self._generate_with_rag(
                    gen_model, current_input_ids, tokenizer, temperature
                )
                path_trace.extend(rag_log)

                code_score_modifier = 0.0
                is_correct = True
                correction_prompt = None

                if tokenizer and self.enable_code_verification:
                    codes = self._extract_code_blocks(current_text)
                    if codes:
                        success, out = self.sandbox.run(codes[-1])
                        feedback = f"Output: {out.strip()}"
                        path_trace.append(f"💻 Code: {codes[-1][:30]}... -> {feedback[:50]}")

                        if not success or "Error" in out:
                            is_correct = False
                            correction_prompt = f"\nSystem: Code execution error: {out}. Rewrite the code correctly.\n"
                            code_score_modifier = -0.5
                        else:
                            code_score_modifier = 0.5
                            result_ids = tokenizer.encode(f"\nResult: {out}\n", return_tensors='pt').to(self.device)
                            generated_ids = torch.cat([generated_ids, result_ids], dim=1)

                with torch.no_grad():
                    if hasattr(self.model, 'embedding'):
                        hidden = self.model.embedding(generated_ids)
                        score = self.verifier(hidden).item() + code_score_modifier
                    else:
                        score = 0.5

                if is_correct or attempt == self.max_retries:
                    candidates.append({
                        "output_ids": generated_ids,
                        "score": score,
                        "trace": path_trace,
                        "strategy": f"neuro_symbolic (attempt {attempt+1})"
                    })
                    break
                else:
                    if correction_prompt:
                        corr_ids = tokenizer.encode(correction_prompt, return_tensors='pt').to(self.device)
                        current_input_ids = torch.cat([generated_ids, corr_ids], dim=1)

        if not candidates:
             return self._system1_inference(input_ids) # Fallback

        best = max(candidates, key=lambda x: x["score"])
        final_text = tokenizer.decode(best["output_ids"][0], skip_special_tokens=True) if tokenizer else ""

        return {
            "final_output": best["output_ids"],
            "final_text": final_text,
            "thought_trace": best["trace"],
            "verifier_score": best["score"],
            "strategy": best["strategy"]
        }

    def _generate_with_rag(
        self, model: Any, input_ids: torch.Tensor, tokenizer: Any, temperature: float
    ) -> Tuple[torch.Tensor, List[str], str]:
        current_ids = input_ids.clone()
        rag_log: List[str] = []

        if tokenizer is None:
            return current_ids, [], ""

        for _ in range(2):
            with torch.no_grad():
                output_ids = model.generate(
                    current_ids,
                    max_length=current_ids.shape[1] + self.max_thinking_steps,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            gen_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            query = self._extract_query(gen_text)

            if query and self.rag_system and self.enable_rag_verification:
                if any(f"Query: {query}" in log for log in rag_log):
                    break
                results = self.rag_system.search(query, k=1)
                knowledge = results[0] if results else "No info."
                rag_log.append(f"Query: {query} -> {knowledge[:30]}...")
                obs_ids = tokenizer.encode(f"\n<obs>{knowledge}</obs>\n", return_tensors='pt').to(self.device)
                current_ids = torch.cat([output_ids, obs_ids], dim=1)
            else:
                current_ids = output_ids
                break

        full_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        return current_ids, rag_log, full_text

    def _system1_inference(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 32,
            do_sample=True,
            repetition_penalty=1.2
        )
        return {"final_output": output, "strategy": "system1_intuition", "thought_trace": [], "verifier_score": 0.0}