"""
matsushibadenki/snn/SNN-122020968b889dca130bffecfd4164ec2223a98b/scripts/experiments/systems/run_unified_mission.py
統合ミッション実行スクリプト (Improved V2)
目的: 統合脳モデル(System 1+2)を用いたエージェントによる、ノイズ環境下での自律ミッション遂行。
      CNNの深化、ノイズ注入学習、LRスケジューラ導入により、精度とエネルギー効率を向上。
"""

import sys
import os
import time
import logging
import random
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore

# プロジェクトルートの設定
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)
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
# SpikingJellyのログを抑制
logging.getLogger("spikingjelly").setLevel(logging.ERROR)

# CuPy関連のエラー抑制を試みる
os.environ["SPIKINGJELLY_NO_CUPY_WARNING"] = "1"

try:
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
    from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


# --- 1. 統合脳モデル (Unified Brain) ---

class VisualCortex(nn.Module):
    """
    視覚野: CNNベースの特徴抽出器 (Enhanced V2)。
    より深い層構造とSiLU活性化関数を採用し、特徴抽出能力を向上させることで
    System 1の精度を高め、System 2への依存度(エネルギー消費)を下げる。
    """
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        # Layer 1: 28x28 -> 14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2: 14x14 -> 7x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Layer 3: 7x7 -> 3x3 (New)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.SiLU()  # Swish activation for better gradient flow
        
        # 最終的な特徴次元への射影
        # 128 * 3 * 3 = 1152
        self.fc = nn.Linear(128 * 3 * 3, feature_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        # Layer 2
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        # Layer 3
        x = self.pool(self.act(self.bn3(self.conv3(x))))
        
        # Flatten & Project
        x = x.flatten(1)
        x = self.dropout(x)
        features = self.act(self.fc(x))  # (B, feature_dim)
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
        
        self.s2_head: Optional[nn.Linear] = None 
        
        # ゲート機構 (Uncertainty Estimation)
        self.gating_net = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, image: torch.Tensor, force_system2: bool = False) -> Dict[str, Any]:
        # 1. 見る (Shared Vision)
        features = self.visual_cortex(image)  # (B, 64)
        
        # 2. 直感 (System 1)
        logits1 = self.system1(features)
        
        # 3. 判断 (Gating)
        uncertainty_map = self.gating_net(features)
        uncertainty_scalar = uncertainty_map.mean().item()
        
        final_logits = logits1
        system_used = "S1"
        
        # System 2 起動閾値 (VisualCortex強化に伴い、少し厳しめに設定して省エネ化)
        threshold = 0.65
        
        # 4. 熟考 (System 2)
        if uncertainty_scalar > threshold or force_system2:
            system_used = "S2"
            
            token_logits = self.feature_to_token(features)
            token_ids = torch.argmax(token_logits, dim=-1).unsqueeze(1)
            
            out2 = self.system2(token_ids)
            if isinstance(out2, tuple):
                out2 = out2[0]
            sys2_feats = out2.mean(dim=1)
            
            if self.s2_head is None:
                feat_dim = sys2_feats.shape[-1]
                self.s2_head = nn.Linear(feat_dim, 10).to(self.device)
            
            logits2 = self.s2_head(sys2_feats)
            
            # アンサンブル: System 2が動いたなら、その意見を尊重する
            final_logits = (logits1 + logits2 * 2.0) / 3.0
            
        return {
            "logits": final_logits,
            "features": features, 
            "system": system_used,
            "uncertainty": uncertainty_scalar,
            "uncertainty_map": uncertainty_map,
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
        self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=self.lr, weight_decay=1e-4)
        # Cosine Annealing Scheduler for better convergence
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.distill_loss = nn.KLDivLoss(reduction="batchmean")
        self.gating_loss_fn = nn.MSELoss()
        
        self.fatigue = 0.0
        self.accuracy_history: list[int] = []

    def set_learning_rate(self, lr: float):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # Schedulerのリセット (新しいフェーズ用)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.1
        )

    def process_data(self, image: torch.Tensor) -> Dict[str, Any]:
        self.brain.eval()
        start_time = time.time()
        with torch.no_grad():
            result = self.brain(image, force_system2=False)
        result["latency"] = (time.time() - start_time) * 1000
        return result

    def learn(self, image: torch.Tensor, label: Optional[torch.Tensor], 
              peer_logits: Optional[torch.Tensor] = None, 
              confidence_weight: float = 1.0):
        self.brain.train()
        self.optimizer.zero_grad()
        
        # Robustness Training: 確率的にノイズを注入して学習する
        # これにより、Mission Phaseのノイズ環境下でのSystem 1精度が向上し、
        # 結果としてSystem 2の起動回数(エネルギー)が減る
        if random.random() < 0.2:
            noise = torch.randn_like(image) * 0.3
            train_image = image + noise
        else:
            train_image = image

        result = self.brain(train_image)
        my_logits = result["logits"]
        uncertainty_map = result["uncertainty_map"]
        
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if label is not None:
            loss = loss + self.criterion(my_logits, label)
            
            with torch.no_grad():
                probs = F.softmax(my_logits, dim=-1)
                p_true = probs.gather(1, label.view(-1, 1))
                gating_target = 1.0 - p_true
            
            loss = loss + self.gating_loss_fn(uncertainty_map, gating_target)

        # 2. OOD Training (Stronger noise)
        noise_input = torch.randn_like(image)
        noise_res = self.brain(noise_input)
        noise_unc = noise_res["uncertainty_map"]
        ood_loss = self.gating_loss_fn(noise_unc, torch.ones_like(noise_unc))
        
        loss = loss + 0.5 * ood_loss
            
        if peer_logits is not None:
            T = 2.0 
            teacher_probs = F.softmax(peer_logits / T, dim=-1)
            my_log_probs = F.log_softmax(my_logits / T, dim=-1)
            distill_scale = 5.0 * confidence_weight
            loss = loss + self.distill_loss(my_log_probs, teacher_probs) * (T**2) * distill_scale
            
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # 疲労蓄積ロジック
        self.fatigue += 0.05
        if result["system"] == "S2":
            self.fatigue += 0.15

        pred = torch.argmax(my_logits, dim=-1)
        return pred

    def sleep_if_tired(self):
        if self.fatigue >= 1.0:
            self.fatigue = 0.0
            return True
        return False

    def update_stats(self, pred: int, label: int):
        self.accuracy_history.append(1 if pred == label else 0)
        if len(self.accuracy_history) > 100:
            self.accuracy_history.pop(0)

    @property
    def current_accuracy(self) -> float:
        if not self.accuracy_history:
            return 0.0
        return sum(self.accuracy_history) / len(self.accuracy_history) * 100


# --- 3. ミッションコントローラー ---

class UnifiedMission:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print("=" * 60)
        print(f"🌌 PROJECT OMEGA: AGI Prototype Initialization (High-Efficiency Mod)")
        print(f"📍 Device: {self.device}")
        print("=" * 60)
        
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
        
        # 学習前にスケジューラをセット
        agent.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.optimizer, T_max=batches, eta_min=1e-6
        )
        
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
        print("\n📚 [Phase 0] Academy Training Phase (w/ Noise Robustness)...")
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
            
            # 異常発生シミュレーション (ノイズ)
            is_anomaly_truth = (random.random() < 0.3)
            if is_anomaly_truth:
                noise = torch.randn_like(image) * 1.5 
                image_input = image + noise
                event_log = "⚠️ ANOMALY"
            else:
                image_input = image
                event_log = "   Normal"
                
            # --- Processing ---
            res_alpha = self.commander.process_data(image_input)
            pred_alpha = torch.argmax(res_alpha["logits"], dim=-1).item()
            unc_alpha = res_alpha["uncertainty"]
            sys_alpha = res_alpha["system"]
            
            res_beta = self.scout.process_data(image_input)
            pred_beta = torch.argmax(res_beta["logits"], dim=-1).item()
            unc_beta = res_beta["uncertainty"]
            sys_beta = res_beta["system"]
            
            alpha_status = ""
            beta_action = ""
            
            # --- Anomaly Detection ---
            alpha_th = 0.7
            beta_th = 0.75
            
            alpha_is_confused = unc_alpha > alpha_th
            beta_is_confused = unc_beta > beta_th
            
            if alpha_is_confused:
                alpha_status = "⚠️ (Skip)"
            
            if beta_is_confused:
                beta_action = "🛡️ (Hold)"
            
            # 学習実行判定
            if not alpha_is_confused and not is_anomaly_truth:
                _ = self.commander.learn(image_input, label)

            # --- Cooperative Logic (v2.9: Safety Protocol) ---
            alpha_is_expert = (unc_alpha < 0.05)
            alpha_is_better = (unc_alpha < 0.2) and (unc_alpha < unc_beta)
            alpha_can_teach = (alpha_is_expert or alpha_is_better) and (not alpha_is_confused)
            
            if alpha_is_confused and not beta_is_confused and (unc_beta > 0.4):
                beta_action = "🛡️ (Cmdr Hold)"
                event_log += " -> Safety Override"
            
            elif not beta_is_confused:
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
                    if unc_beta < 0.3:
                        if not is_anomaly_truth: 
                            self.scout.learn(image_input, label, None)
                            beta_action = "🧠 (Self)"
                        else:
                            beta_action = "🧠 (Self?)" 
                    else:
                        if not beta_action: beta_action = "👀 (Observe)"

            # 統計更新 (Normalのみ)
            if not is_anomaly_truth:
                self.commander.update_stats(pred_alpha, label_val)
                self.scout.update_stats(pred_beta, label_val)
            
            alpha_str = f"{pred_alpha} [{sys_alpha}|u:{unc_alpha:.2f}] {alpha_status}"
            beta_str = f"{pred_beta} [{sys_beta}|u:{unc_beta:.2f}] {beta_action}"
            alpha_mark = "✅" if pred_alpha == label_val else "❌"
            beta_mark = "✅" if pred_beta == label_val else "❌"
            
            if is_anomaly_truth:
                alpha_mark = "🌀"
                beta_mark = "🌀"

            print(f"{step:<4} | {label_val:<6} | {alpha_mark} {alpha_str:<22} | {beta_mark} {beta_str:<22} | {event_log}")
            
            self.commander.sleep_if_tired()
            self.scout.sleep_if_tired()
            time.sleep(0.02)

        print("-" * 85)
        print("🏁 Mission Complete.")
        print(f"   👮‍♂️ Alpha Accuracy (Normal): {self.commander.current_accuracy:.1f}%")
        print(f"   🕵️‍♂️ Beta Accuracy (Normal):  {self.scout.current_accuracy:.1f}%")


if __name__ == "__main__":
    mission = UnifiedMission()
    mission.run_mission(steps=40)