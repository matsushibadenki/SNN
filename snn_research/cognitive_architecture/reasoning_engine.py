# ファイルパス: snn_research/cognitive_architecture/reasoning_engine.py
# 日本語タイトル: Reasoning Engine v2.0 - Program-Aided Verification & System 2 Loop
# 目的・内容:
#   ROADMAP v16.1/v17 "Thinking & Verifier" の完全実装。
#   既存のニューラル検証に加え、Pythonコード生成・実行による論理的検証(Program-Aided Verification)を追加。
#   思考パス(CoT)の中にコードブロックが含まれていた場合、それを実行して解の正当性を確かめる。

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, cast, Tuple
import logging
import re
import io
import contextlib
import multiprocessing
import traceback
import sys

# プロジェクト内依存関係
from snn_research.models.transformer.sformer import SFormer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

logger = logging.getLogger(__name__)

# --- Helper: Safe Code Execution ---

class CodeSandbox:
    """
    生成されたPythonコードを安全に実行するためのサンドボックス環境。
    ロードマップ "7. 開発ルール" に基づき、推論をコードで裏付けする。
    """
    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout
        # 危険な操作を行うモジュールのインポートを禁止
        self.forbidden_modules = [
            "os", "sys", "subprocess", "shutil", "netrc", "requests", "urllib", "socket", "pathlib"
        ]

    def _execute_script(self, code: str, result_queue: multiprocessing.Queue):
        """別プロセスで実行される関数"""
        # 1. 安全性チェック (簡易静的解析)
        for mod in self.forbidden_modules:
            if re.search(fr"\b(import|from)\s+{mod}\b", code):
                result_queue.put(("Error", f"Security Violation: Import of '{mod}' is forbidden."))
                return
        
        if "open(" in code or "exec(" in code or "eval(" in code:
             result_queue.put(("Error", "Security Violation: Usage of 'open', 'exec', or 'eval' is forbidden."))
             return

        # 2. 実行環境の構築
        f = io.StringIO()
        
        # 実行結果を格納する変数
        local_scope: Dict[str, Any] = {}
        
        try:
            with contextlib.redirect_stdout(f):
                # 実行 (__builtins__ を制限して安全性を高めることも可能だが、ここでは利便性のため標準を使用)
                # ただしファイル操作等はforbidden_modulesチェックで弾く
                exec(code, {"__name__": "__main__", "__builtins__": __builtins__}, local_scope)
            
            output = f.getvalue().strip()
            
            # 出力がない場合、最後の式の結果を取得しようとする努力（REPL的な挙動）はここではしない。
            # Print出力を正とする。
            result_queue.put(("Success", output))
            
        except Exception:
            # エラー発生時はトレースバックを返す
            result_queue.put(("Error", traceback.format_exc()))

    def run(self, code: str) -> Tuple[bool, str]:
        """
        コードを実行し、(成功フラグ, 出力またはエラー) を返す。
        タイムアウト付きのプロセス分離実行を行う。
        """
        queue: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=self._execute_script, args=(code, queue))
        p.start()
        
        p.join(self.timeout)
        
        if p.is_alive():
            p.terminate()
            p.join()
            return False, "Timeout: Code execution took too long."
        
        if not queue.empty():
            status, output = queue.get()
            return (status == "Success"), output
        else:
            # キューが空のまま終了した場合（クラッシュなど）
            return False, "Unknown Execution Error (Process crashed or returned nothing)"


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
        # 平均プーリングで文章全体の特徴を集約（将来的にはAttention Pooling推奨）
        pooled = torch.mean(hidden_states, dim=1)
        compressed = self.compressor(pooled)
        score = self.value_head(compressed)
        return score


class ReasoningEngine:
    """
    熟慮（Thinking）と検証（Verifier）のループを制御するエンジン。
    System 2 として機能し、直感的な回答だけでなく、
    論理的推論やコード実行による検証を行う。
    """
    def __init__(
        self,
        generative_model: SFormer,
        astrocyte: AstrocyteNetwork,
        verifier_model: Optional[VerifierNetwork] = None,
        d_model: int = 256,
        num_thinking_paths: int = 4, 
        max_thinking_steps: int = 64, # ステップ数をロードマップに合わせて増加
        enable_code_verification: bool = True,
        device: str = 'cpu'
    ):
        self.model = generative_model
        self.astrocyte = astrocyte
        self.device = device
        self.enable_code_verification = enable_code_verification
        
        if verifier_model is None:
            self.verifier = VerifierNetwork(d_model=d_model).to(device)
        else:
            self.verifier = verifier_model.to(device)
            
        self.num_thinking_paths = num_thinking_paths
        self.max_thinking_steps = max_thinking_steps
        
        # コードサンドボックスの初期化
        self.sandbox = CodeSandbox(timeout=2.0)
        
        logger.info(f"🧠 ReasoningEngine v2 initialized (Paths: {num_thinking_paths}, CodeVerify: {enable_code_verification}, Device: {device})")

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Markdown形式のコードブロックを抽出する"""
        # ```python ... ``` または ``` ... ``` を抽出
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches if m.strip()]

    def think_and_solve(
        self, 
        input_ids: torch.Tensor, 
        task_type: str = "general",
        temperature: float = 0.7,
        tokenizer: Any = None  # コード検証のためにトークナイザを受け取れるように拡張
    ) -> Dict[str, Any]:
        """
        Thinking Loop: 
        1. 複数の思考パスを生成 (CoT)
        2. Neural Verifierで初期評価
        3. (Optional) コードブロックがあれば実行して検証 (Program-Aided Verification)
        4. 最良の思考パスを選択 (Best-of-N)
        """
        B = input_ids.shape[0]
        if B != 1:
            # 現状の実装簡略化のためバッチサイズ1に限定
            raise ValueError("ReasoningEngine currently supports batch_size=1 only.")

        # 1. エネルギーリソース確認 (Astrocyte)
        # 思考は高コストなため、リソースを確認する
        estimated_cost = self.num_thinking_paths * self.max_thinking_steps * 0.15
        module_name = "prefrontal_cortex" if task_type == "logic" else "cortex"
        
        if not self.astrocyte.request_resource(module_name, estimated_cost):
            logger.info("⚡ Low Energy: ReasoningEngine fallback to System 1 (Intuition).")
            return self._system1_inference(input_ids)

        # 2. System 2: Thinking Loop
        logger.info(f"🤔 System 2 Activated: Generating {self.num_thinking_paths} paths with temp={temperature}")
        
        candidates = []
        batch_input = input_ids.repeat(self.num_thinking_paths, 1) # (N, L)
        
        self.model.eval()
        self.verifier.eval()
        
        with torch.no_grad():
            # 2.1 思考パスの並列生成 (Parallel Generation)
            # SFormerのgenerateメソッドを使用
            gen_model = cast(Any, self.model)
            
            if not hasattr(gen_model, 'generate'):
                logger.warning("Generative model does not have 'generate' method. Fallback.")
                return self._system1_inference(input_ids)

            generated_ids = gen_model.generate(
                batch_input, 
                max_length=self.max_thinking_steps + input_ids.shape[1],
                temperature=temperature,
                do_sample=True
            ) # (N, TotalLen)

            # 2.2 Neural Verifier による評価 (初期スコア)
            # Embedding層を通して隠れ状態を近似 (SFormerの出力層直前などが理想だが簡易実装)
            hidden_approx = self.model.embedding(generated_ids) 
            neural_scores = self.verifier(hidden_approx).view(-1).tolist()
            
            # 候補リストの初期化
            for i in range(self.num_thinking_paths):
                candidates.append({
                    "output_ids": generated_ids[i],
                    "score": neural_scores[i],
                    "path_id": i,
                    "code_feedback": None,
                    "strategy": "neural_only"
                })

        # 3. Program-Aided Verification (コード実行検証)
        # トークナイザが提供され、かつコード検証が有効な場合
        if tokenizer is not None and self.enable_code_verification:
            logger.debug("🔎 Running Program-Aided Verification...")
            
            input_len = input_ids.shape[1]
            
            for candidate in candidates:
                # 生成部分のみをデコード
                gen_ids = candidate["output_ids"][input_len:]
                generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                
                code_blocks = self._extract_code_blocks(generated_text)
                
                if code_blocks:
                    # 最後のコードブロックを実行対象とみなす（解答コードと仮定）
                    code_to_run = code_blocks[-1]
                    success, output = self.sandbox.run(code_to_run)
                    
                    candidate["code_feedback"] = f"Result: {'Success' if success else 'Fail'}\nOutput: {output[:100]}..."
                    
                    # スコアの再調整 (Re-ranking Logic)
                    if success:
                        # 実行成功かつ出力がある場合、信頼度をブースト
                        # ただし出力がエラーメッセージっぽくないことも要確認だがここでは簡易的に
                        if "Error" not in output and "Exception" not in output:
                            candidate["score"] = min(1.0, candidate["score"] + 0.25)
                            candidate["strategy"] = "code_verified_success"
                        else:
                             candidate["score"] = max(0.0, candidate["score"] - 0.1)
                    else:
                        # 実行時エラーは減点
                        candidate["score"] = max(0.0, candidate["score"] - 0.25)
                        candidate["strategy"] = "code_verified_failed"

        # 4. Selection (Best-of-N)
        # スコアが最も高いものを選択
        best_candidate = max(candidates, key=lambda x: x["score"])
        
        # 思考トレースの構築 (デバッグ・可視化用)
        trace = []
        for c in candidates:
            trace_info = f"Path {c['path_id']}: Score={c['score']:.3f} [{c['strategy']}]"
            if c['code_feedback']:
                trace_info += f" | Code: {c['code_feedback']}"
            trace.append(trace_info)

        return {
            "final_output": best_candidate["output_ids"].unsqueeze(0),
            "thought_trace": trace,
            "verifier_score": best_candidate["score"],
            "strategy": best_candidate["strategy"]
        }

    def _system1_inference(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """System 1: 即時推論 (Greedy) - 低エネルギーモード"""
        self.model.eval()
        gen_model = cast(Any, self.model)
        
        with torch.no_grad():
            if hasattr(gen_model, 'generate'):
                output = gen_model.generate(
                    input_ids, 
                    max_length=input_ids.shape[1] + 16,
                    do_sample=False # Greedy
                )
            else:
                logits, _, _ = self.model(input_ids)
                output = torch.argmax(logits, dim=-1)
                
        return {
            "final_output": output,
            "strategy": "system1_fast",
            "thought_trace": ["System 1: Direct Intuition"],
            "verifier_score": 0.0 # System 1は検証なし
        }
