# ファイルパス: scripts/run_distillation_experiment.py
# Title: Chain-of-Thought (CoT) Distillation Runner
# Description:
#   ROADMAP v16.5 実装。
#   System 2 (ReasoningEngine) の「思考プロセス」を教師データとして生成し、
#   System 1 (SFormer Small) に蒸留する実験を行うスクリプト。
#   これにより、推論時の計算コストを大幅に削減（省エネ化）する。

import sys
import os
import torch
import logging
import argparse
from transformers import AutoTokenizer # type: ignore

# プロジェクトルートの設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("DistillationExp")

from snn_research.core.snn_core import SNNCore
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.distill.pipeline import AdvancedDistillationPipeline

def main():
    parser = argparse.ArgumentParser(description="Run CoT Distillation Experiment")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_prompts", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"🚀 Starting Distillation Experiment on {device}")

    # 1. コンポーネントの準備
    logger.info("1. Initializing Teacher (System 2) & Student (System 1)...")
    
    # Tokenizer (GPT-2 for demo)
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Tokenizer load failed: {e}")
        return

    # --- Teacher: Reasoning Engine (Large SFormer) ---
    # ※デモ用にパラメータは小さめに設定しています
    teacher_config = {
        "architecture_type": "sformer",
        "d_model": 128, 
        "num_layers": 4, 
        "n_head": 4,
        "neuron_config": {"base_threshold": 1.0}
    }
    teacher_core = SNNCore(teacher_config, vocab_size=tokenizer.vocab_size).to(device)
    astrocyte = AstrocyteNetwork()
    
    # 教師は「考える」ことができる
    teacher_engine = ReasoningEngine(
        generative_model=teacher_core.model, # type: ignore
        astrocyte=astrocyte,
        device=device,
        num_thinking_paths=2, # 複数の思考パスを試す
        max_thinking_steps=10
    )

    # --- Student: Lightweight SFormer (Small) ---
    student_config = {
        "architecture_type": "sformer",
        "d_model": 64,  # 教師より小さい
        "num_layers": 2, # 層も少ない
        "n_head": 2,
        "neuron_config": {"base_threshold": 0.5} # 発火しやすい
    }
    student_core = SNNCore(student_config, vocab_size=tokenizer.vocab_size).to(device)
    
    # 2. パイプラインの構築
    pipeline = AdvancedDistillationPipeline(
        student_model=student_core.model, # type: ignore
        device=device,
        tokenizer=tokenizer
    )

    # 3. プロンプトの準備 (タスク例: 論理推論)
    logger.info("2. Generating Reasoning Traces (Rollout)...")
    prompts = [
        "Problem: If A is bigger than B, and B is bigger than C, is A bigger than C? Answer:",
        "Problem: What happens if you drop a glass on concrete? Answer:",
        "Problem: 2 + 2 * 2 = ? Answer:",
        # ... 本来はデータセットからロード
    ] * (args.num_prompts // 3 + 1)
    prompts = prompts[:args.num_prompts]

    # 4. 蒸留実行 (CoT Distillation)
    # Teacherが思考を行い、そのプロセスをStudentが学習する
    logger.info("3. Running CoT Distillation...")
    pipeline.run_cot_distillation(
        reasoning_engine=teacher_engine,
        prompts=prompts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-4
    )

    # 5. 評価 (Before/Afterの比較は本来必要だが、ここでは動作確認)
    logger.info("4. Verifying Student Capability...")
    test_input = "Problem: 2 + 2 * 2 = ? Answer:"
    input_ids = tokenizer.encode(test_input, return_tensors='pt').to(device)
    
    student_core.model.eval()
    with torch.no_grad():
        # SFormer.generate が実装されている前提
        if hasattr(student_core.model, 'generate'):
            output_ids = student_core.model.generate(input_ids, max_length=20, do_sample=False) # type: ignore
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logger.info(f"🎓 Student Output: {output_text}")
        else:
            logger.warning("Student model does not support generation.")

    logger.info("✅ Distillation Experiment Completed.")

if __name__ == "__main__":
    main()