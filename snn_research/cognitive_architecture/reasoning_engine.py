# ファイルパス: snn_research/cognitive_architecture/reasoning_engine.py
# 日本語タイトル: Reasoning Engine v2.4 - String Input Support & Auto-Tokenization
# 目的・内容:
#   System 2 Engine の完全実装。
#   - AsyncBrainKernelからの文字列入力に対応するため、Tokenizerを統合。
#   - process() メソッドでの型判定と変換ロジックを強化。

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, cast, Tuple
import logging
import re
import io
import contextlib
import multiprocessing
from transformers import PreTrainedTokenizerBase

# 依存関係
from snn_research.models.transformer.sformer import SFormer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.rag_snn import RAGSystem

logger = logging.getLogger(__name__)

class CodeSandbox:
    """生成されたPythonコードを安全に実行するためのサンドボックス環境"""
    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout
        self.forbidden_modules = [
            "os", "sys", "subprocess", "shutil", "netrc", "requests", "urllib", 
            "socket", "pathlib", "input", "open"
        ]

    def _execute_code(self, code: str, queue: multiprocessing.Queue):
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
    """
    熟慮（Thinking）と検証（Verifier）、そして外部知識検索（RAG）を統合したSystem 2エンジン。
    """
    def __init__(
        self,
        generative_model: SFormer,
        astrocyte: AstrocyteNetwork,
        tokenizer: Optional[PreTrainedTokenizerBase] = None, # 追加
        rag_system: Optional[RAGSystem] = None,
        verifier_model: Optional[VerifierNetwork] = None,
        d_model: int = 256,
        num_thinking_paths: int = 3,
        max_thinking_steps: int = 128, 
        enable_code_verification: bool = True,
        enable_rag_verification: bool = True,
        max_retries: int = 2,
        device: str = 'cpu'
    ):
        self.model = generative_model
        self.astrocyte = astrocyte
        self.tokenizer = tokenizer # 追加
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
        self.sandbox = CodeSandbox(timeout=2.0)
        
        logger.info(f"🧠 ReasoningEngine v2.4 initialized (Tokenizer: {tokenizer is not None}).")

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        AsyncBrainKernelからのエントリポイント。文字列、辞書、Tensorに対応。
        """
        input_ids = None
        current_tokenizer = self.tokenizer
        
        try:
            # 1. 文字列入力の場合
            if isinstance(input_data, str):
                if self.tokenizer is None:
                    return {"error": "Tokenizer required for string input", "strategy": "none"}
                
                encoded = self.tokenizer(input_data, return_tensors="pt")
                input_ids = encoded.input_ids.to(self.device)
            
            # 2. 辞書型の場合
            elif isinstance(input_data, dict):
                if "input_ids" in input_data:
                    input_ids = input_data["input_ids"]
                    if isinstance(input_ids, torch.Tensor):
                        input_ids = input_ids.to(self.device)
                elif "text" in input_data and self.tokenizer:
                    encoded = self.tokenizer(input_data["text"], return_tensors="pt")
                    input_ids = encoded.input_ids.to(self.device)
                
                # ペイロード内のTokenizerを優先
                if "tokenizer" in input_data:
                    current_tokenizer = input_data["tokenizer"]

            # 3. Tensorの場合
            elif isinstance(input_data, torch.Tensor):
                input_ids = input_data.to(self.device)
            
            if input_ids is None:
                return {"error": "Could not extract input_ids", "strategy": "none"}

            return self.think_and_solve(
                input_ids=input_ids,
                tokenizer=current_tokenizer
            )
        except Exception as e:
            logger.error(f"ReasoningEngine Process Error: {e}", exc_info=True)
            return {"error": str(e), "strategy": "error_recovery"}

    def _extract_query(self, text: str) -> Optional[str]:
        pattern = r"<query>(.*?)</query>"
        match = re.search(pattern, text, re.DOTALL)
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
        B = input_ids.shape[0]
        if B != 1: 
            return self._system1_inference(input_ids)

        estimated_cost = self.num_thinking_paths * self.max_thinking_steps * 0.2
        if not self.astrocyte.request_resource("prefrontal_cortex", estimated_cost):
            logger.info("⚡ Low Energy: Fallback to System 1.")
            return self._system1_inference(input_ids)

        logger.info(f"🤔 System 2 Activated: Generating {self.num_thinking_paths} paths...")
        
        candidates = []
        gen_model = cast(Any, self.model)
        
        for path_idx in range(self.num_thinking_paths):
            current_input_ids = input_ids.clone()
            path_trace: List[str] = []
            final_code_feedback = None
            
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
                        final_code_feedback = f"{'Success' if success else 'Fail'}: {out[:100]}"
                        if not success or "Error" in out:
                            is_correct = False
                            correction_prompt = f"\nExecution Error: {out}. Please fix.\n"
                            code_score_modifier = -0.5
                        else:
                            code_score_modifier = 0.5
                            path_trace.append(f"Code Executed: {out[:50]}")

                if is_correct or attempt == self.max_retries:
                    with torch.no_grad():
                        if hasattr(self.model, 'embedding'):
                            hidden_approx = self.model.embedding(generated_ids)
                            score = self.verifier(hidden_approx).item() + code_score_modifier
                        else:
                            score = 0.5 
                    
                    candidates.append({
                        "output_ids": generated_ids,
                        "score": score,
                        "trace": path_trace,
                        "strategy": f"neuro_symbolic (attempt {attempt+1})"
                    })
                    break
                else:
                    logger.info(f"   🔄 Path {path_idx} Correction: {final_code_feedback}")
                    if correction_prompt and tokenizer:
                        correction_ids = tokenizer.encode(correction_prompt, return_tensors='pt').to(self.device)
                        current_input_ids = torch.cat([generated_ids, correction_ids], dim=1)

        if not candidates:
            return self._system1_inference(input_ids)

        best = max(candidates, key=lambda x: x["score"])
        
        # デコードしてテキストも返す
        final_text = ""
        if tokenizer:
             final_text = tokenizer.decode(best["output_ids"][0], skip_special_tokens=True)

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
             return current_ids, ["Error: No tokenizer"], ""

        for _ in range(3):
            with torch.no_grad():
                output_ids = model.generate(
                    current_ids, 
                    max_length=current_ids.shape[1] + (self.max_thinking_steps // 2),
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            query = self._extract_query(generated_text)
            
            if query and self.rag_system and self.enable_rag_verification:
                if any(f"Query: {query}" in log for log in rag_log):
                    current_ids = output_ids
                    break

                logger.info(f"   🔍 RAG Query: '{query}'")
                results = self.rag_system.search(query, k=2)
                knowledge = "\n".join(results) if results else "No relevant info."
                rag_log.append(f"Query: {query} -> Obs: {knowledge[:30]}...")
                
                obs_ids = tokenizer.encode(f"\n<observation>{knowledge}</observation>\n", return_tensors='pt').to(self.device)
                current_ids = torch.cat([output_ids, obs_ids], dim=1)
                continue 
            else:
                current_ids = output_ids
                break
        
        full_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        return current_ids, rag_log, full_text

    def _system1_inference(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        gen_model = cast(Any, self.model)
        output = gen_model.generate(input_ids, max_length=input_ids.shape[1] + 32, do_sample=False)
        return {"final_output": output, "strategy": "system1_fallback", "thought_trace": [], "verifier_score": 0.0}