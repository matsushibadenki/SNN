# ファイルパス: scripts/experiments/systems/run_phase7_civilization.py
# 日本語タイトル: Phase 7 デジタル文明シミュレーション "Eden" v1.2 (Loss Fix)
# 目的: 複数のAGIプロトタイプ(Genesis)が相互作用し、知識を共有・継承する社会システムの構築。
# 修正履歴:
#   v1.2: CrossEntropyLossの入力型エラーを修正 (Logitsを返すように変更)。

import sys
import os
import time
import logging
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional

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
# 外部ライブラリのログ抑制
logging.getLogger("spikingjelly").setLevel(logging.ERROR)

# 必要なモジュールのインポート
try:
    from snn_research.core.snn_core import SNNCore
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
    from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


class CivilizationBrain(nn.Module):
    """
    文明用ハイブリッド脳。他者との通信機能を持つ。
    """
    def __init__(self, device: str, vocab_size: int = 128):
        super().__init__()
        self.device = device
        
        # System 1: SFormer
        self.system1 = SNNCore(config={
            "architecture_type": "sformer",
            "d_model": 64,
            "num_layers": 2,
            "nhead": 2,
            "time_steps": 2,
            "neuron_config": {"type": "lif", "v_threshold": 1.0}
        }, vocab_size=vocab_size).to(device)
        
        # System 2: BitSpikeMamba
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
        
        # 意思決定層
        self.decision_layer = nn.Linear(vocab_size, 3).to(device)
        
        # 知識エンコーダ
        self.speech_layer = nn.Linear(vocab_size, vocab_size).to(device)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        if not x.is_contiguous(): x = x.contiguous()
        
        # System 1
        out1 = self.system1(x)
        if isinstance(out1, tuple): out1 = out1[0]
        features = out1.mean(dim=1) # (Batch, Vocab)
        
        # System 2
        if context is not None:
            out2 = self.system2(x) 
            if isinstance(out2, tuple): out2 = out2[0]
            features = (features + out2.mean(dim=1)) / 2.0
            
        action_logits = self.decision_layer(features)
        speech_logits = self.speech_layer(features)
        
        return {
            "action": torch.argmax(action_logits, dim=-1),
            "speech": torch.argmax(speech_logits, dim=-1),
            "speech_logits": speech_logits, # [Fix] Logitsを返す
            "features": features
        }


class Citizen:
    """
    デジタル文明の市民エージェント。
    """
    def __init__(self, name: str, device: str, generation: int = 1):
        self.name = name
        self.device = device
        self.generation = generation
        
        self.brain = CivilizationBrain(device).to(device)
        self.sleep_system = SleepConsolidator(target_brain_model=self.brain.system2)
        self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=0.002)
        
        self.knowledge_score = 0
        self.fatigue = 0.0
        self.social_bond = 0
        
    def act(self, env_input: torch.Tensor, peer_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        self.brain.eval()
        with torch.no_grad():
            result = self.brain(env_input, context=peer_input)
        return result

    def learn_from_peer(self, peer_speech: torch.Tensor):
        """他者の発話から学ぶ (模倣学習/知識伝達)"""
        self.brain.train()
        self.optimizer.zero_grad()
        
        # 相手の言葉を聞いて、自分も同じ概念を想起できるか (Autoencoder的)
        # peer_speechは (1,) のスカラーテンソル(トークンID)
        
        # 入力: (Batch=1, Seq=1) に整形
        dummy_input = peer_speech.unsqueeze(0) 
        if dummy_input.dim() == 1:
             dummy_input = dummy_input.unsqueeze(0)

        result = self.brain(dummy_input)
        
        # [Fix] CrossEntropyLossには (Batch, Class) のLogitsと (Batch) のTargetを渡す
        logits = result["speech_logits"] # (1, Vocab)
        target = peer_speech             # (1)
        
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        self.optimizer.step()
        
        self.knowledge_score += 1
        self.fatigue += 0.1
        return loss.item()

    def rest(self):
        """休息と記憶の整理"""
        if self.fatigue > 0.5:
            summary = self.sleep_system.perform_sleep_cycle(duration_cycles=1)
            self.fatigue = 0.0
            return True
        return False


class EdenSimulation:
    def __init__(self, population_size: int = 4):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🚀 Initializing Phase 7 Civilization 'Eden' on {self.device}...")
        
        names = ["Adam", "Eve", "Cain", "Abel", "Seth", "Mary", "Noah", "Lilith"]
        self.population = [
            Citizen(names[i % len(names)], self.device) for i in range(population_size)
        ]
        self.year = 0

    def run_year(self):
        self.year += 1
        print(f"\n🌍 Year {self.year}: The sun rises on Eden.")
        
        # 環境からの刺激 (日替わりの「謎」)
        daily_mystery = torch.randint(0, 128, (1, 8)).to(self.device)
        
        # 全市民の行動
        interactions = []
        
        for citizen in self.population:
            # 1. 休息チェック
            if citizen.rest():
                print(f"   💤 {citizen.name} is sleeping. Dreaming of electric sheep...")
                continue
                
            # 2. 行動選択
            result = citizen.act(daily_mystery)
            action = result["action"].item() # 0:Explore, 1:Talk, 2:Rest
            speech = result["speech"] # 発話内容(トークン)
            
            # アクション実行
            if action == 0: # 探索
                citizen.knowledge_score += 1
                citizen.fatigue += 0.1
                # 独り言をつぶやく (思考)
                pass 
                
            elif action == 1: # 対話 (他者を探す)
                # ランダムに相手を選ぶ
                partner = random.choice([c for c in self.population if c != citizen])
                interactions.append((citizen, partner, speech))
                
            elif action == 2: # 休息
                citizen.fatigue += 0.05 # 待機疲れ
        
        # 3. 社会的相互作用の解決
        for actor, partner, speech in interactions:
            # パートナーが起きているか確認
            if partner.fatigue < 0.8:
                loss = partner.learn_from_peer(speech)
                actor.social_bond += 1
                partner.social_bond += 1
                print(f"   🗣️ {actor.name} shared wisdom with {partner.name}. (Loss: {loss:.4f})")
            else:
                print(f"   🚫 {actor.name} tried to talk, but {partner.name} was too tired.")

        # 4. 統計表示
        avg_knowledge = sum(c.knowledge_score for c in self.population) / len(self.population)
        print(f"   📊 Avg Knowledge: {avg_knowledge:.1f} | Active Interactions: {len(interactions)}")

    def evolve(self):
        """世代交代: 知識の少ない個体は淘汰され、優秀な個体が増える"""
        if self.year % 10 == 0:
            print("\n⚡ Evolution Event triggered!")
            # 知識スコアでソート
            sorted_pop = sorted(self.population, key=lambda x: x.knowledge_score, reverse=True)
            
            # 上位半分が生き残り、複製される
            survivors = sorted_pop[:len(sorted_pop)//2]
            new_generation = []
            
            for parent in survivors:
                # 親
                new_generation.append(parent)
                # 子 (親の脳パラメータを継承 + 変異は今回はなし)
                child_name = f"{parent.name}_Jr"
                child = Citizen(child_name, self.device, generation=parent.generation + 1)
                child.brain.load_state_dict(parent.brain.state_dict()) # 知識継承
                new_generation.append(child)
                print(f"   👶 {parent.name} passed knowledge to {child_name} (Gen {child.generation})")
                
            self.population = new_generation

    def run_simulation(self, years: int = 50):
        try:
            for _ in range(years):
                self.run_year()
                self.evolve()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user.")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    eden = EdenSimulation(population_size=6)
    eden.run_simulation(years=50)