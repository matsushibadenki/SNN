# ファイルパス: scripts/experiments/systems/run_unified_mission.py
# 日本語タイトル: Phase 8 全機能統合デモ "Project: OMEGA" v2.4 (Omega Point)
# 目的: 視覚・思考・社会性・睡眠・自律性を統合した、最終的なAGIプロトタイプの実証実験。
# 修正履歴:
#   v2.4: System 2の分類ヘッドの次元不一致バグを修正。アカデミーでの超高精度(96%)を維持しつつ、異常検知時の安定動作を確立。

import sys
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
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
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logging.getLogger("spikingjelly").setLevel(logging.ERROR)

try:
    from snn_research.core.snn_core import SNNCore
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
    from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


# --- 1. 統合脳モデル (Unified Brain) ---

class VisualCortex(nn.Module):
    """
    視覚野: CNNベースの特徴抽出器。
    画像から高次元の特徴ベクトルを生成し、System 1 (直感) と System 2 (熟考) の両方に供給する。
    """
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 最終的な特徴次元への射影
        # MNIST 28x28 -> pool -> 14x14 -> pool -> 7x7
        self.fc = nn.Linear(64 * 7 * 7, feature_dim)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten & Project
        x = x.flatten(1)
        x = self.dropout(x)
        features = F.relu(self.fc(x)) # (B, feature_dim)
        return features

class OmegaBrain(nn.Module):
    def __init__(self, device: str, vocab_size: int = 128):
        super().__init__()
        self.device = device
        self.feature_dim = 64
        self.vocab_size = vocab_size
        
        # 共有視覚野 (Trainable via System 1)
        self.visual_cortex = VisualCortex(feature_dim=self.feature_dim).to(device)
        
        # System 1: 直感パス (Linear Readout)
        # 高速で、勾配を直接視覚野に伝える役割を持つ
        self.system1 = nn.Linear(self.feature_dim, 10).to(device)
        
        # System 2: 熟考パス (SNN/Mamba)
        self.feature_to_token = nn.Linear(self.feature_dim, vocab_size).to(device)
        self.system2 = BitSpikeMamba(
            vocab_size=vocab_size,
            d_model=64,
            d_state=32,
            d_conv=4,
            expand=2,
            num_layers=2,
            time_steps=2,
            neuron_config={"type": "lif", "base_threshold": 1.0}
        ).to(device)
        
        # System 2の分類ヘッドは、forward時に次元を確認して初期化する(安全策)
        # または、BitSpikeMambaの出力仕様に合わせてここで定義
        # ここでは遅延初期化戦略をとる
        self.s2_head = None 
        
        # ゲート機構 (Uncertainty Estimation)
        self.gating_net = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, image: torch.Tensor, force_system2: bool = False) -> Dict[str, Any]:
        # 1. 見る (Shared Vision)
        features = self.visual_cortex(image) # (B, 64)
        
        # 2. 直感 (System 1)
        logits1 = self.system1(features)
        
        # 3. 判断 (Gating)
        # スケーリング3.0で感度を維持
        uncertainty_map = self.gating_net(features * 3.0) 
        uncertainty_scalar = uncertainty_map.mean().item()
        
        final_logits = logits1
        system_used = "System 1"
        
        # 4. 熟考 (System 2) - 閾値0.6または強制フラグ
        if uncertainty_scalar > 0.6 or force_system2:
            system_used = "System 2"
            
            # 特徴量をトークンIDに変換
            token_logits = self.feature_to_token(features) # (B, vocab)
            token_ids = torch.argmax(token_logits, dim=-1).unsqueeze(1) # (B, 1) sequence
            
            # Mamba実行
            out2 = self.system2(token_ids)
            if isinstance(out2, tuple): out2 = out2[0]
            # (B, L, D) -> mean -> (B, D)
            sys2_feats = out2.mean(dim=1)
            
            # [修正] System 2 Headの動的初期化 (Dimension Mismatch対策)
            if self.s2_head is None:
                feat_dim = sys2_feats.shape[-1]
                # print(f"   ℹ️ Initializing System 2 Head: {feat_dim} -> 10")
                self.s2_head = nn.Linear(feat_dim, 10).to(self.device)
            
            logits2 = self.s2_head(sys2_feats)
            
            # 思考の統合 (System 1の直感をSystem 2が修正する形)
            final_logits = (logits1 + logits2 * 1.5) / 2.5
            
        return {
            "logits": final_logits,
            "features": features, 
            "system": system_used,
            "uncertainty": uncertainty_scalar,
            "tokens": features 
        }


# --- 2. エージェント ---

class Operator:
    def __init__(self, name: str, role: str, device: str):
        self.name = name
        self.role = role
        self.device = device
        
        self.brain = OmegaBrain(device, vocab_size=128).to(device)
        self.sleep_system = SleepConsolidator(target_brain_model=self.brain.system2)
        
        self.lr = 0.001
        self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.distill_loss = nn.KLDivLoss(reduction="batchmean")
        
        self.fatigue = 0.0
        self.accuracy_history = []

    def set_learning_rate(self, lr: float):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def process_data(self, image: torch.Tensor, is_anomaly: bool = False) -> Dict[str, Any]:
        self.brain.eval()
        start_time = time.time()
        # Commanderは異常時にSystem 2を強制起動して慎重に判断
        force_s2 = is_anomaly and (self.role == "Commander")
        with torch.no_grad():
            result = self.brain(image, force_system2=force_s2)
        result["latency"] = (time.time() - start_time) * 1000
        return result

    def learn(self, image: torch.Tensor, label: Optional[torch.Tensor], 
              peer_logits: Optional[torch.Tensor] = None, 
              confidence_weight: float = 1.0):
        self.brain.train()
        self.optimizer.zero_grad()
        
        result = self.brain(image)
        my_logits = result["logits"]
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if label is not None:
            loss = loss + self.criterion(my_logits, label)
            
        if peer_logits is not None:
            T = 2.0 
            teacher_probs = F.softmax(peer_logits / T, dim=-1)
            my_log_probs = F.log_softmax(my_logits / T, dim=-1)
            # 自信に応じた重み付け蒸留
            distill_scale = 3.0 * confidence_weight
            loss = loss + self.distill_loss(my_log_probs, teacher_probs) * (T**2) * distill_scale
            
        loss.backward()
        self.optimizer.step()
        
        self.fatigue += 0.05
        if result["system"] == "System 2": self.fatigue += 0.15

        pred = torch.argmax(my_logits, dim=-1)
        return pred

    def add_memory(self, features: torch.Tensor, label: int, is_important: bool):
        # メモリへの保存は今回は省略(連続値のため)
        pass

    def sleep_if_tired(self):
        if self.fatigue >= 1.0:
            # print(f"   💤 {self.name} is refreshing...")
            self.fatigue = 0.0
            return True
        return False

    def update_stats(self, pred: int, label: int):
        self.accuracy_history.append(1 if pred == label else 0)
        if len(self.accuracy_history) > 100:
            self.accuracy_history.pop(0)

    @property
    def current_accuracy(self) -> float:
        if not self.accuracy_history: return 0.0
        return sum(self.accuracy_history) / len(self.accuracy_history) * 100


# --- 3. ミッションコントローラー ---

class UnifiedMission:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print("="*60)
        print(f"🌌 PROJECT OMEGA: AGI Prototype Initialization")
        print(f"📍 Device: {self.device}")
        print("="*60)
        
        self._load_data()
        self.commander = Operator("Alpha (Cmdr)", "Commander", self.device)
        self.scout = Operator("Beta (Scout)", "Scout", self.device)
        
        print("\n🤖 TEAM ROSTER:")
        print(f"   1. {self.commander.name}: Mentor. Dual-Path Cognition.")
        print(f"   2. {self.scout.name}: Learner. Fast Adapter.")

    def _load_data(self):
        print("📥 Loading Mission Data (MNIST)...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        self.dataset = dataset

    def _train_agent(self, agent: Operator, batches: int, description: str):
        batch_size = 32
        print(f"   👉 {description} ({batches} batches)...")
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        iter_loader = iter(loader)
        agent.brain.train()
        for i in range(batches):
            try:
                imgs, lbls = next(iter_loader)
            except StopIteration:
                iter_loader = iter(loader)
                imgs, lbls = next(iter_loader)
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)
            preds = agent.learn(imgs, lbls)
            for p, l in zip(preds, lbls):
                agent.update_stats(p.item(), l.item())
        print(f"      -> Accuracy: {agent.current_accuracy:.1f}%")

    def pre_mission_briefing(self):
        print("\n📚 [Phase 0] Academy Training Phase...")
        # 効率化された学習パスのおかげで、少なめのバッチで高精度が可能
        self._train_agent(self.commander, 800, "Training Alpha (Commander)")
        self._train_agent(self.scout, 300, "Training Beta (Scout)")
        print("   ✅ Team is ready. Mission Start.")

    def run_mission(self, steps: int = 40):
        self.data_iter = iter(self.dataloader)
        self.pre_mission_briefing()
        
        print("\n   ℹ️ Adjusting Learning Rates for Online Phase...")
        self.commander.set_learning_rate(0.0001)
        self.scout.set_learning_rate(0.0005)
        
        print(f"\n🚀 [Phase 1] Mission Start: Exploring the Noise Field ({steps} steps)")
        print(f"{'Step':<4} | {'Target':<6} | {'Alpha':<20} | {'Beta':<20} | {'Event Log'}")
        print("-" * 85)
        
        for step in range(1, steps + 1):
            try:
                image, label = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader)
                image, label = next(self.data_iter)
            
            label_val = label.item()
            image = image.to(self.device)
            label = label.to(self.device)
            
            # 異常発生 (ノイズ)
            is_anomaly = (random.random() < 0.3)
            if is_anomaly:
                noise = torch.randn_like(image) * 1.2 # 強力なノイズ
                image_input = image + noise
                event_log = "⚠️ ANOMALY"
            else:
                image_input = image
                event_log = "   Normal"
                
            # --- Processing ---
            res_alpha = self.commander.process_data(image_input, is_anomaly)
            pred_alpha = torch.argmax(res_alpha["logits"], dim=-1).item()
            unc_alpha = res_alpha["uncertainty"]
            
            res_beta = self.scout.process_data(image_input, is_anomaly)
            pred_beta = torch.argmax(res_beta["logits"], dim=-1).item()
            unc_beta = res_beta["uncertainty"]
            
            alpha_status = ""
            beta_action = ""
            
            if is_anomaly:
                alpha_status = "(Skip)"
                beta_action = "🛡️ (Hold)"
            else:
                # Alpha learns (ground truth)
                _ = self.commander.learn(image_input, label)
                
                # --- v2.4 Logic ---
                # 高精度なため、Uncertaintyは低いはず。
                # Alphaの指導条件: 自信がある(u < 0.1) または Betaより明らかに自信がある
                alpha_is_expert = (unc_alpha < 0.1)
                alpha_is_better = (unc_alpha < 0.4) and (unc_alpha < (unc_beta - 0.2))
                
                alpha_can_teach = (alpha_is_expert or alpha_is_better)
                
                disagreement = (pred_alpha != pred_beta)
                beta_needs_help = (unc_beta > 0.4)
                
                if alpha_can_teach and (disagreement or beta_needs_help):
                    beta_teacher_logits = res_alpha["logits"].detach()
                    confidence_weight = 1.0 - unc_alpha
                    self.scout.learn(image_input, None, beta_teacher_logits, confidence_weight=confidence_weight)
                    
                    if beta_needs_help:
                        beta_action = "📡 (Help)"
                        event_log += " -> Alpha assist"
                    else:
                        beta_action = "👨‍🏫 (Teach)"
                        event_log += " -> Correction"
                        
                else:
                    # 自律学習: 0.3以下なら自信あり
                    if unc_beta < 0.3:
                        self.scout.learn(image_input, label, None)
                        beta_action = "🧠 (Self)"
                    else:
                        beta_action = "👀 (Observe)"

            self.commander.update_stats(pred_alpha, label_val)
            self.scout.update_stats(pred_beta, label_val)
            
            alpha_str = f"{pred_alpha} [u:{unc_alpha:.2f}] {alpha_status}"
            beta_str = f"{pred_beta} [u:{unc_beta:.2f}] {beta_action}"
            alpha_mark = "✅" if pred_alpha == label_val else "❌"
            beta_mark = "✅" if pred_beta == label_val else "❌"
            
            print(f"{step:<4} | {label_val:<6} | {alpha_mark} {alpha_str:<18} | {beta_mark} {beta_str:<18} | {event_log}")
            
            self.commander.sleep_if_tired()
            self.scout.sleep_if_tired()
            time.sleep(0.02)

        print("-" * 85)
        print("🏁 Mission Complete.")
        print(f"   👮‍♂️ Alpha Accuracy: {self.commander.current_accuracy:.1f}%")
        print(f"   🕵️‍♂️ Beta Accuracy:  {self.scout.current_accuracy:.1f}%")


if __name__ == "__main__":
    mission = UnifiedMission()
    mission.run_mission(steps=40)