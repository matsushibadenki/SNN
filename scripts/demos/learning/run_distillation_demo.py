# ファイルパス: scripts/runners/run_distillation_demo.py
# Title: System 1/2 Distillation Demo
# Description:
#   Symbolic Teacher (System 2) が生成した算数の解法プロセスを、
#   BitSpikeStudent (System 1) に蒸留し、思考能力の獲得をシミュレーションする。

import sys
import os
import torch
import torch.nn as nn
import logging
import random

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from snn_research.distillation.thought_distiller import ThoughtDistillationManager, SymbolicTeacher

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("DistillationDemo")

# --- Mock Student Model (BitSpikeMamba Placeholder) ---
class MockBitSpikeStudent(nn.Module):
    """
    デモ用の簡易BitSpikeモデル。
    実際にはここに BitSpikeMamba などの実モデルが入る。
    """
    def __init__(self):
        super().__init__()
        # 学習可能なパラメータ (ダミー)
        self.weights = nn.Parameter(torch.randn(10, 10))
        self.knowledge_level = 0.0 # 0.0 (Baka) -> 1.0 (Genius)
    
    def forward_text_loss(self, input_text, target_text):
        """
        テキスト入力を受け取り、Lossを返すインターフェースのモック。
        学習が進むにつれてLossが下がる挙動をシミュレート。
        """
        # 簡易シミュレーション: 学習が進むとLossが下がる
        current_loss = max(2.0 - self.knowledge_level * 2.0, 0.1)
        # ランダムな揺らぎ
        noise = random.uniform(-0.1, 0.1)
        
        # 学習効果の蓄積 (本来はOptimizerが行うが、ここでは簡略化)
        self.knowledge_level += 0.05
        
        return torch.tensor(current_loss + noise, requires_grad=True)

    def generate(self, input_text):
        """推論"""
        if self.knowledge_level < 0.3:
            return "I don't know... maybe 10?"
        elif self.knowledge_level < 0.8:
            return "First add ones... something... Answer: ??"
        else:
            # 教師のロジックを模倣したような出力を返す
            return "First, add ones... Write 2, carry 1... Answer: Correct!"

def main():
    print("⚗️ --- System 1/2 Distillation Demo ---")
    print("   Target: Distill arithmetic reasoning into BitSpike Model")
    
    # 1. コンポーネント初期化
    student = MockBitSpikeStudent()
    teacher = SymbolicTeacher()
    manager = ThoughtDistillationManager(student, teacher)
    
    # 2. 問題セットの作成
    problems = [
        "15 + 27", "12 + 19", "35 + 46", "8 + 5", "50 + 50",
        "22 + 39", "9 + 91", "14 + 16", "77 + 3", "25 + 25"
    ]
    
    # 3. 教師による思考データの生成 (System 2 Generation)
    print("\n🔹 Phase 1: Teacher (System 2) Thought Generation")
    dataset = manager.generate_thought_dataset(problems)
    
    # データの例を表示
    sample = dataset[0]
    print(f"   Sample Data:")
    print(f"   Q: {sample['input']}")
    print(f"   CoT: {sample['thought_chain']}")
    print(f"   A: {sample['answer']}")
    
    # 4. 蒸留前の生徒の性能確認
    print("\n🔹 Phase 2: Pre-Distillation Student Performance")
    test_q = "Q: 15 + 27"
    print(f"   Student says: {student.generate(test_q)}")
    
    # 5. 蒸留学習 (Distillation)
    print("\n🔹 Phase 3: Distillation Training")
    manager.distill(dataset, epochs=5)
    
    # 6. 蒸留後の生徒の性能確認
    print("\n🔹 Phase 4: Post-Distillation Student Performance")
    print(f"   Student says: {student.generate(test_q)}")
    
    if "Correct" in student.generate(test_q):
        print("\n✅ SUCCESS: System 1 successfully internalized System 2's reasoning!")
    else:
        print("\n⚠️ PARTIAL: Training complete, but mimicking needs improvement.")

if __name__ == "__main__":
    main()