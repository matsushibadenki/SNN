# ファイルパス: scripts/experiments/social/run_phase5_social_learning.py
# 日本語タイトル: Phase 5 社会的学習エージェント v1.3 (Distillation Boost)
# 目的: 2体のエージェント(先生と生徒)が知識を共有し、集合知効果を実証する。
# 修正履歴:
#   v1.1: バッチサイズ増加、モデル軽量化。
#   v1.2: Classifier次元修正。
#   v1.3: Teacher予習機能追加、学習率調整(0.002)、蒸留重み強化(10.0)。

import sys
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# プロジェクトルートの設定
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
# 不要なログの抑制
logging.getLogger("spikingjelly").setLevel(logging.ERROR)
logging.getLogger("VisualAgent").setLevel(logging.ERROR)

# 必要なモジュールのインポート
try:
    from snn_research.core.snn_core import SNNCore
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


# --- 脳モデルの定義 (軽量化版) ---

class VisualTokenizer(nn.Module):
    def __init__(self, vocab_size: int = 128, patch_size: int = 4):
        super().__init__()
        self.patch_conv = nn.Conv2d(1, vocab_size, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous(): x = x.contiguous()
        features = self.patch_conv(x)
        features = features.flatten(2).transpose(1, 2).contiguous()
        visual_tokens = torch.argmax(features, dim=-1)
        return visual_tokens

class SocialBrain(nn.Module):
    def __init__(self, device: str, vocab_size: int = 128, name: str = "Brain"):
        super().__init__()
        self.device = device
        self.name = name
        
        self.visual_cortex = VisualTokenizer(vocab_size=vocab_size, patch_size=4).to(device)
        
        sformer_config = {
            "architecture_type": "sformer",
            "d_model": 64,
            "num_layers": 2,
            "nhead": 2,
            "time_steps": 2,
            "neuron_config": {"type": "lif", "v_threshold": 1.0}
        }
        self.core = SNNCore(config=sformer_config, vocab_size=vocab_size).to(device)
        self.classifier = nn.Linear(vocab_size, 10).to(device)

    def forward(self, image: torch.Tensor) -> Dict[str, Any]:
        visual_tokens = self.visual_cortex(image)
        core_out = self.core(visual_tokens)
        
        if isinstance(core_out, tuple): core_out = core_out[0]
        
        features = core_out.mean(dim=1)
        logits = self.classifier(features)
        
        return {"logits": logits, "features": features}


class SocialAgent:
    def __init__(self, name: str, role: str, device: str):
        self.name = name
        self.role = role 
        self.device = device
        self.brain = SocialBrain(device, name=name, vocab_size=128).to(device)
        
        # [調整] 学習率を少し下げて安定化
        self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=0.002)
        self.criterion = nn.CrossEntropyLoss()
        self.distill_loss = nn.KLDivLoss(reduction="batchmean")
        self.correct_history = [] 

    def perceive_and_act(self, image: torch.Tensor) -> Dict[str, Any]:
        self.brain.eval()
        with torch.no_grad():
            result = self.brain(image)
        return result

    def learn(self, image: torch.Tensor, label: torch.Tensor, teacher_logits: Optional[torch.Tensor] = None) -> float:
        self.brain.train()
        self.optimizer.zero_grad()
        
        result = self.brain(image)
        student_logits = result["logits"]
        
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if label is not None:
            loss = loss + self.criterion(student_logits, label)
            
        if teacher_logits is not None and self.role == "Student":
            temperature = 2.5 # [調整] 温度パラメータ
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            
            distill_loss_val = self.distill_loss(student_log_probs, teacher_probs) * (temperature ** 2)
            loss = loss + distill_loss_val * 10.0 # [調整] 蒸留の影響をさらに強化
            
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_metrics(self, predictions: torch.Tensor, labels: torch.Tensor):
        if labels is not None:
            corrects = (predictions == labels).float().tolist()
            self.correct_history.extend(corrects)
            if len(self.correct_history) > 100:
                self.correct_history = self.correct_history[-100:]

    @property
    def accuracy(self) -> float:
        if not self.correct_history:
            return 0.0
        return sum(self.correct_history) / len(self.correct_history) * 100


class SocialExperiment:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🚀 Initializing Phase 5 Social Experiment (Boosted) on {self.device}...")
        
        self._prepare_data()
        
        self.teacher = SocialAgent("Alice", "Teacher", self.device)
        self.student = SocialAgent("Bob",   "Student", self.device)
        
        print("   👩‍🏫 Agent Alice (Teacher): Full Label Access.")
        print("   🧑‍🎓 Agent Bob   (Student): Label Access 10% (Relies on Alice).")

    def _prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True) 

    def pretrain_teacher(self, steps: int = 50):
        """先生役(Alice)を事前に少し賢くしておく"""
        print(f"\n📚 Pre-training Teacher (Alice) for {steps} steps...")
        iter_data = iter(self.dataloader)
        for _ in range(steps):
            try:
                images, labels = next(iter_data)
            except StopIteration:
                iter_data = iter(self.dataloader)
                images, labels = next(iter_data)
            
            images, labels = images.to(self.device), labels.to(self.device)
            self.teacher.learn(images, labels)
            
            out = self.teacher.perceive_and_act(images)
            pred = torch.argmax(out["logits"], dim=-1)
            self.teacher.update_metrics(pred, labels)
            
        print(f"   -> Alice Initial Accuracy: {self.teacher.accuracy:.2f}%")

    def run(self, cycles: int = 300):
        # 実験前にAliceを予習させる
        self.pretrain_teacher(steps=50)
        
        print("\n🎬 Starting Social Learning Loop (300 Batches)...")
        print(f"{'Batch':<6} | {'Alice Acc':<12} | {'Bob Acc':<12} | {'Status':<20}")
        print("-" * 60)
        
        iter_data = iter(self.dataloader)
        
        for step in range(1, cycles + 1):
            try:
                images, labels = next(iter_data)
            except StopIteration:
                iter_data = iter(self.dataloader)
                images, labels = next(iter_data)
                
            images, labels = images.to(self.device), labels.to(self.device)
            
            # --- 1. Alice (Teacher) ---
            alice_out = self.teacher.perceive_and_act(images)
            self.teacher.learn(images, labels)
            
            alice_pred = torch.argmax(alice_out["logits"], dim=-1)
            self.teacher.update_metrics(alice_pred, labels)
            
            # --- 2. Bob (Student) ---
            has_label_access = (np.random.random() < 0.1)
            
            bob_out = self.student.perceive_and_act(images)
            
            status = ""
            if has_label_access:
                self.student.learn(images, labels)
                status = "Bob: Self-Study 📖"
            else:
                self.student.learn(images, None, teacher_logits=alice_out["logits"].detach())
                status = "Bob: Asked Alice 🗣️"

            bob_pred = torch.argmax(bob_out["logits"], dim=-1)
            self.student.update_metrics(bob_pred, labels)

            if step % 10 == 0:
                print(f"{step:<6} | {self.teacher.accuracy:6.1f}%      | {self.student.accuracy:6.1f}%      | {status}")
                
            time.sleep(0.01)

        print("-" * 60)
        print("📊 Final Results:")
        print(f"   👩‍🏫 Alice (Teacher) Accuracy: {self.teacher.accuracy:.2f}%")
        print(f"   🧑‍🎓 Bob   (Student) Accuracy: {self.student.accuracy:.2f}%")
        
        if self.student.accuracy > 30.0:
            print("✅ Collective Intelligence Confirmed! Bob learned significantly via communication.")
        elif self.student.accuracy > 15.0:
            print("⚠️ Bob learned slightly, but needs more time.")
        else:
            print("❌ Learning failed. Check model configuration.")

if __name__ == "__main__":
    try:
        experiment = SocialExperiment()
        experiment.run(cycles=300) 
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()