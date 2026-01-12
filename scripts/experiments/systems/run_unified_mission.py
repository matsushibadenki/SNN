# ファイルパス: scripts/experiments/systems/run_unified_mission.py
# 日本語タイトル: Phase 8 全機能統合デモ "Project: OMEGA" v1.1
# 目的: 視覚・思考・社会性・睡眠・自律性を統合した、最終的なAGIプロトタイプの実証実験。
# 修正履歴:
#   v1.1: Pre-training時のバッチサイズ(32)に対応するため、Gating判定を .mean().item() に修正。

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

# ロギング設定 (リッチな出力)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logging.getLogger("spikingjelly").setLevel(logging.ERROR)

# 必要なモジュールのインポート
try:
    from snn_research.core.snn_core import SNNCore
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
    from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


# --- 1. 統合脳モデル (Unified Brain) ---

class VisualTokenizer(nn.Module):
    """視覚野: 画像を脳が理解できるトークン列に変換"""
    def __init__(self, vocab_size: int = 128, patch_size: int = 4):
        super().__init__()
        self.patch_conv = nn.Conv2d(1, vocab_size, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous(): x = x.contiguous()
        features = self.patch_conv(x) # (B, C, H, W)
        features = features.flatten(2).transpose(1, 2).contiguous() # (B, L, C)
        visual_tokens = torch.argmax(features, dim=-1) # 量子化
        return visual_tokens

class OmegaBrain(nn.Module):
    """
    Project OMEGAのための統合脳。
    System 1 (直感/SFormer) と System 2 (熟考/Mamba) を搭載。
    """
    def __init__(self, device: str, vocab_size: int = 128):
        super().__init__()
        self.device = device
        
        # 視覚入力
        self.visual_cortex = VisualTokenizer(vocab_size=vocab_size, patch_size=4).to(device)
        
        # System 1: SFormer (Fast, Low Energy)
        self.system1 = SNNCore(config={
            "architecture_type": "sformer",
            "d_model": 64,
            "num_layers": 2,
            "nhead": 2,
            "time_steps": 2,
            "neuron_config": {"type": "lif", "v_threshold": 1.0}
        }, vocab_size=vocab_size).to(device)
        
        # System 2: BitSpikeMamba (Slow, Deep, High Energy)
        self.system2 = BitSpikeMamba(
            vocab_size=vocab_size,
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
            num_layers=2,
            time_steps=4,
            neuron_config={"type": "lif", "base_threshold": 1.0}
        ).to(device)
        
        # ゲート機構: System 1の出力の「曖昧さ」を監視
        self.gating_net = nn.Sequential(
            nn.Linear(vocab_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(device)
        
        # 出力層 (数字分類 0-9)
        self.classifier = nn.Linear(vocab_size, 10).to(device)

    def forward(self, image: torch.Tensor, force_system2: bool = False) -> Dict[str, Any]:
        # 1. 見る (Visual Cortex)
        tokens = self.visual_cortex(image)
        
        # 2. 直感で考える (System 1)
        out1 = self.system1(tokens)
        if isinstance(out1, tuple): out1 = out1[0]
        sys1_feats = out1.mean(dim=1) # (B, Vocab)
        
        # 3. 判断する (Gating)
        uncertainty_map = self.gating_net(sys1_feats)
        
        # [Fix] バッチサイズ > 1 の場合に対応するため .mean() を使用
        uncertainty_scalar = uncertainty_map.mean().item()
        
        final_feats = sys1_feats
        system_used = "System 1"
        
        # 閾値を超えるか、強制フラグがあればSystem 2起動
        if uncertainty_scalar > 0.6 or force_system2:
            system_used = "System 2"
            out2 = self.system2(tokens)
            if isinstance(out2, tuple): out2 = out2[0]
            sys2_feats = out2.mean(dim=1)
            
            # 思考の統合
            final_feats = (sys1_feats + sys2_feats) / 2.0
            
        # 4. 答えを出す
        logits = self.classifier(final_feats)
        
        return {
            "logits": logits,
            "features": final_feats,
            "system": system_used,
            "uncertainty": uncertainty_scalar,
            "tokens": tokens # 記憶用
        }


# --- 2. エージェント (The Operators) ---

class Operator:
    def __init__(self, name: str, role: str, device: str):
        self.name = name
        self.role = role # "Commander" (Teacher) or "Scout" (Student)
        self.device = device
        
        self.brain = OmegaBrain(device).to(device)
        self.sleep_system = SleepConsolidator(target_brain_model=self.brain.system2)
        
        # 学習設定
        self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=0.002)
        self.criterion = nn.CrossEntropyLoss()
        self.distill_loss = nn.KLDivLoss(reduction="batchmean")
        
        self.fatigue = 0.0
        self.experience_buffer = []
        self.accuracy_history = []

    def process_data(self, image: torch.Tensor, is_anomaly: bool = False) -> Dict[str, Any]:
        """環境データを処理し、思考する"""
        self.brain.eval()
        start_time = time.time()
        
        # 異常検知時は慎重になる (System 2強制)
        force_s2 = is_anomaly and (self.role == "Commander")
        
        with torch.no_grad():
            result = self.brain(image, force_system2=force_s2)
            
        latency = (time.time() - start_time) * 1000
        result["latency"] = latency
        return result

    def learn(self, image: torch.Tensor, label: Optional[torch.Tensor], peer_logits: Optional[torch.Tensor] = None):
        """学習フェーズ: 経験または他者から学ぶ"""
        self.brain.train()
        self.optimizer.zero_grad()
        
        result = self.brain(image)
        my_logits = result["logits"]
        
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 教師あり学習 (Commanderは常に可能, Scoutは稀)
        if label is not None:
            loss = loss + self.criterion(my_logits, label)
            
        # 社会的学習 (ScoutがCommanderから学ぶ)
        if peer_logits is not None and self.role == "Scout":
            T = 3.0
            teacher_probs = F.softmax(peer_logits / T, dim=-1)
            my_log_probs = F.log_softmax(my_logits / T, dim=-1)
            loss = loss + self.distill_loss(my_log_probs, teacher_probs) * (T**2) * 5.0
            
        loss.backward()
        self.optimizer.step()
        
        # 疲労蓄積
        self.fatigue += 0.05
        if result["system"] == "System 2":
            self.fatigue += 0.15 # 深く考えると疲れる

    def add_memory(self, tokens: torch.Tensor, label: int, is_important: bool):
        """重要なイベントを海馬へ"""
        if is_important:
            mem_tokens = tokens.cpu()
            mem_label = torch.tensor([label]).cpu()
            self.sleep_system.store_experience(mem_tokens, mem_label, 1.0)

    def sleep_if_tired(self):
        """疲労したら眠る"""
        if self.fatigue >= 1.0:
            print(f"   💤 {self.name} is entering Deep Sleep cycle...")
            summary = self.sleep_system.perform_sleep_cycle(duration_cycles=2)
            consolidated = summary.get('consolidated_to_cortex', 0)
            print(f"      -> {self.name} consolidated {consolidated} memories. Brain optimized.")
            self.fatigue = 0.0
            return True
        return False

    def update_stats(self, pred: int, label: int):
        self.accuracy_history.append(1 if pred == label else 0)
        if len(self.accuracy_history) > 50:
            self.accuracy_history.pop(0)

    @property
    def current_accuracy(self) -> float:
        if not self.accuracy_history: return 0.0
        return sum(self.accuracy_history) / len(self.accuracy_history) * 100


# --- 3. ミッションコントローラー (Environment) ---

class UnifiedMission:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print("="*60)
        print(f"🌌 PROJECT OMEGA: AGI Prototype Initialization")
        print(f"📍 Device: {self.device}")
        print("="*60)
        
        self._load_data()
        
        # エージェント生成
        self.commander = Operator("Alpha (Cmdr)", "Commander", self.device)
        self.scout = Operator("Beta (Scout)", "Scout", self.device)
        
        print("\n🤖 TEAM ROSTER:")
        print(f"   1. {self.commander.name}: High Spec, Full Access. Uses System 2.")
        print(f"   2. {self.scout.name}: Agile, Learning. Relies on Alpha.")

    def _load_data(self):
        print("📥 Loading Mission Data (MNIST)...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        self.data_iter = iter(self.dataloader)

    def pre_mission_briefing(self):
        """司令官(Alpha)に事前知識を与える"""
        print("\n📚 [Phase 0] Pre-Mission Briefing for Alpha...")
        
        # 短期集中学習
        briefing_steps = 100
        loader = DataLoader(self.dataloader.dataset, batch_size=32, shuffle=True)
        iter_brief = iter(loader)
        
        for _ in range(briefing_steps // 32):
            try:
                imgs, lbls = next(iter_brief)
            except: break
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)
            self.commander.learn(imgs, lbls)
            
        print("   ✅ Alpha is ready. Mission Start.")

    def run_mission(self, steps: int = 30):
        self.pre_mission_briefing()
        
        print(f"\n🚀 [Phase 1] Mission Start: Exploring the Noise Field ({steps} steps)")
        print(f"{'Step':<4} | {'Target':<6} | {'Alpha':<18} | {'Beta':<18} | {'Event Log'}")
        print("-" * 85)
        
        for step in range(1, steps + 1):
            # 1. データの取得
            try:
                image, label = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader)
                image, label = next(self.data_iter)
            
            label_val = label.item()
            image = image.to(self.device)
            label = label.to(self.device)
            
            # 2. 異常発生 (ノイズ)
            is_anomaly = (random.random() < 0.3)
            if is_anomaly:
                noise = torch.randn_like(image) * 0.8
                image = image + noise
                event_log = "⚠️ ANOMALY DETECTED"
            else:
                event_log = "   Normal Scan"
                
            # 3. Alpha (司令官) の行動
            res_alpha = self.commander.process_data(image, is_anomaly)
            pred_alpha = torch.argmax(res_alpha["logits"], dim=-1).item()
            
            # Alphaは常に正解を見て学習し、経験を積む
            self.commander.learn(image, label)
            self.commander.update_stats(pred_alpha, label_val)
            
            # 4. Beta (スカウト) の行動
            res_beta = self.scout.process_data(image, is_anomaly) # Betaは自力で考える
            pred_beta = torch.argmax(res_beta["logits"], dim=-1).item()
            
            # Betaの判断ロジック
            beta_action = ""
            beta_learn_target = None
            beta_teacher_logits = None
            
            # Betaが間違っている、または自信がない(System 2起動など)場合、Alphaに通信
            # (ここではシミュレーションとして、Anomaly時は必ず通信)
            if is_anomaly or res_beta["system"] == "System 2":
                event_log += " -> 📡 Beta requesting backup"
                beta_teacher_logits = res_alpha["logits"].detach()
                beta_action = "(Help)"
                # 重要な経験として記憶
                self.scout.add_memory(res_beta["tokens"], label_val, True)
            else:
                # 平時は自力学習 (正解ラベルへのアクセスは稀: 10%)
                if random.random() < 0.1:
                    beta_learn_target = label
                    beta_action = "(Self)"
            
            # Betaの学習実行
            self.scout.learn(image, beta_learn_target, beta_teacher_logits)
            self.scout.update_stats(pred_beta, label_val)
            
            # 5. 結果表示
            alpha_str = f"{pred_alpha} [{res_alpha['system']:^8}]"
            beta_str = f"{pred_beta} [{res_beta['system']:^8}] {beta_action}"
            
            # 正解判定マーク
            alpha_mark = "✅" if pred_alpha == label_val else "❌"
            beta_mark = "✅" if pred_beta == label_val else "❌"
            
            print(f"{step:<4} | {label_val:<6} | {alpha_mark} {alpha_str} | {beta_mark} {beta_str} | {event_log}")
            
            # 6. 睡眠管理 (自律性)
            self.commander.sleep_if_tired()
            self.scout.sleep_if_tired()
            
            time.sleep(0.1)

        print("-" * 85)
        print("🏁 Mission Complete.")
        print(f"   👮‍♂️ Alpha Accuracy: {self.commander.current_accuracy:.1f}%")
        print(f"   🕵️‍♂️ Beta Accuracy:  {self.scout.current_accuracy:.1f}% (Learned via collaboration)")


if __name__ == "__main__":
    mission = UnifiedMission()
    mission.run_mission(steps=40)