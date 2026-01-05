# ファイルパス: scripts/runners/run_neuro_symbolic_demo.py
# Title: Neuro-Symbolic & Sleep Distillation Demo
# Description:
#   v16.1の全機能を統合したデモンストレーション。
#   1. RAG知識ベースの構築（フェイクデータ）
#   2. ReasoningEngineによる「検索を伴う思考」の実行
#   3. 思考結果のSleepConsolidatorへの蓄積と蒸留（学習）

from snn_research.io.spike_encoder import SpikeEncoder  # ダミー用
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork  # E402 fixed
from snn_research.models.transformer.sformer import SFormer
import sys
import os
import torch
import logging
from transformers import GPT2Tokenizer

# パス設定
project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("NeuroSymDemo")


def main():
    print("🧠 --- Neuro-Symbolic SNN & Sleep Distillation Demo ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Components
    # SFormer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    sformer = SFormer(vocab_size=50257, d_model=128,
                      nhead=4, num_layers=4).to(device)

    # RAG System (Mocking Data)
    rag = RAGSystem(vector_store_path="workspace/runs/demo_store")
    # 知識注入: SNNについての知識を持たせる
    rag.add_document(
        "SNN (Spiking Neural Network) is energy efficient.", metadata={"subj": "SNN"})
    rag.add_triple("SNN", "uses", "spikes")
    rag.add_triple("SNN", "mimics", "brain")

    # Astrocyte & Reasoning
    astrocyte = AstrocyteNetwork()
    reasoning = ReasoningEngine(
        generative_model=sformer,
        astrocyte=astrocyte,
        rag_system=rag,
        enable_rag_verification=True,
        device=device
    )

    # Sleep Consolidator
    spike_encoder = SpikeEncoder()  # ダミー
    sleep_manager = SleepConsolidator(rag, sformer, spike_encoder)

    # 2. Run Reasoning Task (Thinking with RAG)
    print("\n🔹 Phase 1: Reasoning with RAG")
    # ユーザー入力: "Tell me about SNN efficiency. <query>SNN efficiency</query>"
    # ※本来はLLMが<query>を自律生成するが、ここではデモのためプロンプトに埋め込む
    input_text = "Question: What is SNN? <query>SNN</query>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    result = reasoning.think_and_solve(input_ids, tokenizer=tokenizer)

    print("   [Trace]")
    for t in result["thought_trace"]:
        print(f"   - {t}")

    # 生成された思考結果を睡眠マネージャに登録
    sleep_manager.add_thought_trace(result)

    # 3. Run Sleep Cycle (Distillation)
    print("\n🔹 Phase 2: Sleep & Distillation")
    print("   Going to sleep to consolidate the reasoning experience...")

    stats = sleep_manager.perform_sleep_cycle()

    print(f"   Replayed Dreams: {stats['dreams_replayed']}")
    print(f"   Synaptic Change (Loss): {stats['synaptic_change']:.4f}")

    print("\n🎉 Demo Completed: The brain reasoned, learned from tools, and consolidated memory.")


if __name__ == "__main__":
    main()
