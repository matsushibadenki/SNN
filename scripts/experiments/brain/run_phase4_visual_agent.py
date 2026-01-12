# ファイルパス: scripts/experiments/brain/run_phase4_visual_agent.py
# 日本語タイトル: Phase 4 視覚野搭載ハイブリッド・エージェント (Visual Cortex & MNIST)
# 目的: 実際の画像データ(MNIST)を入力とし、視覚トークナイザーを通じてSystem 1/2で認識・学習を行う。

import sys
import os
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
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
    format='%(asctime)s - [VisualAgent] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger("VisualAgent")

# 必要なモジュールのインポート
try:
    from snn_research.core.snn_core import SNNCore
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
    from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
except ImportError as e:
    logger.error(f"❌ Import Error: {e}")
    sys.exit(1)


class VisualTokenizer(nn.Module):
    """
    視覚野 (Visual Cortex) の初期段階。
    画像パッチを処理し、脳が理解できる「視覚単語（Visual Tokens）」に量子化する。
    """
    def __init__(self, vocab_size: int = 1000, patch_size: int = 4):
        super().__init__()
        # MNIST(28x28) -> 4x4パッチ -> 7x7=49トークン
        self.patch_conv = nn.Conv2d(1, vocab_size, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        # MPS対策: メモリ整列
        if not x.is_contiguous():
            x = x.contiguous()
            
        # 特徴抽出: (B, Vocab, 7, 7)
        features = self.patch_conv(x)
        
        # フラット化: (B, Vocab, 49) -> (B, 49, Vocab)
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2).contiguous()
        
        # 量子化: 各パッチで最も反応の強いチャネルをトークンIDとする
        # これにより、SFormer等のEmbedding層に入力可能な形式(LongTensor)になる
        visual_tokens = torch.argmax(features, dim=-1) # (B, 49)
        
        return visual_tokens


class VisualHybridBrain(nn.Module):
    """
    視覚トークンを処理するハイブリッド脳。
    """
    def __init__(self, device: str, vocab_size: int = 1000):
        super().__init__()
        self.device = device
        
        # 1. 視覚野 (Visual Cortex)
        logger.info("   👁️ Initializing Visual Cortex (Tokenizer)...")
        self.visual_cortex = VisualTokenizer(vocab_size=vocab_size, patch_size=4).to(device)
        
        # 2. System 1: SFormer (高速視覚処理)
        logger.info("   🧠 Initializing System 1: SFormer (Visual Reflex)...")
        sformer_config = {
            "architecture_type": "sformer",
            "d_model": 256,
            "num_layers": 2,
            "nhead": 4,
            "time_steps": 4,
            "neuron_config": {"type": "lif", "v_threshold": 1.0}
        }
        # 入力シーケンス長は 7x7=49
        self.system1 = SNNCore(config=sformer_config, vocab_size=vocab_size).to(device)
        
        # 3. System 2: BitSpikeMamba (詳細分析)
        logger.info("   🧠 Initializing System 2: BitSpikeMamba (Visual Reasoning)...")
        self.system2 = BitSpikeMamba(
            vocab_size=vocab_size,
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
            num_layers=4,
            time_steps=8,
            neuron_config={"type": "lif", "base_threshold": 1.0}
        ).to(device)
        
        # 4. 出力層 (数字0-9の分類)
        self.classifier = nn.Linear(256, 10).to(device)
        
        # 5. ゲート機構 (不確実性に基づく切り替え)
        # System 1 の出力(Vocab次元)から判断
        self.gating_network = nn.Sequential(
            nn.Linear(vocab_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, image: torch.Tensor, noise_level: float = 0.0) -> Dict[str, Any]:
        # 視覚野によるトークン化
        visual_tokens = self.visual_cortex(image) # (B, 49)
        
        # System 1 実行
        sys1_feats = self.system1(visual_tokens) # (B, Seq, Vocab)
        if isinstance(sys1_feats, tuple): sys1_feats = sys1_feats[0]
        
        # 特徴量の平均化 (Classification用)
        # ここでは単純化のため、Vocab次元を特徴量として扱う
        sys1_pooled = sys1_feats.mean(dim=1) # (B, Vocab)
        
        # ゲート判断
        gate_score = self.gating_network(sys1_pooled).mean().item()
        
        # ノイズレベルが高い場合や、System 1が自信がない場合はSystem 2を起動
        # (シミュレーションのため、noise_levelも判断に加える)
        use_system2 = gate_score > 0.6 or noise_level > 0.3
        
        used_system = "System 1"
        final_feats = sys1_pooled
        
        if use_system2:
            used_system = "System 2 (Activated)"
            sys2_feats = self.system2(visual_tokens)
            if isinstance(sys2_feats, tuple): sys2_feats = sys2_feats[0]
            
            # System 2の特徴量とSystem 1の特徴量を統合（ここでは単純置換）
            # 次元合わせ: Mamba出力は(B, Seq, D_model=256)想定だが、実装により異なるため調整
            # BitSpikeMambaの出力は (B, L, Vocab)
            
            final_feats = sys2_feats.mean(dim=1) # (B, Vocab)
        
        # 最終分類 (Vocab次元 -> 256へ射影が必要だが、簡易的にVocab次元の一部を使用するか、再射影)
        # ここでは classifier の入力次元(256)に合わせるため、Vocab(1000) -> 256 の射影層を通すか、スライスする
        # 簡易実装: Vocab次元の先頭256を使用
        logits = self.classifier(final_feats[:, :256])
        
        return {
            "logits": logits,
            "system": used_system,
            "gate_score": gate_score,
            "visual_tokens": visual_tokens # 記憶用
        }


class Phase4VisualAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"🚀 Initializing Phase 4 Visual Agent on {self.device}...")
        
        # データセットの準備 (MNIST)
        self._prepare_data()
        
        # 脳の構築
        self.brain = VisualHybridBrain(self.device).to(self.device)
        
        # 睡眠システム (長期記憶)
        self.sleep_system = SleepConsolidator(
            target_brain_model=self.brain.system2
        )
        
        self.fatigue = 0.0
        self.steps = 0

    def _prepare_data(self):
        """MNISTデータセットのロード"""
        logger.info("   📥 Loading MNIST dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # データセットがない場合はダウンロード
        try:
            dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            self.data_iter = iter(self.dataloader)
        except Exception as e:
            logger.warning(f"   ⚠️ Could not load MNIST: {e}. Using dummy noise data.")
            self.dataloader = None

    def get_visual_input(self) -> Tuple[torch.Tensor, int, float]:
        """環境から視覚入力を取得"""
        noise_level = 0.0
        
        if self.dataloader:
            try:
                image, label = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader)
                image, label = next(self.data_iter)
                
            # 時々画像にノイズを加える（難易度アップ -> System 2 起動用）
            if np.random.random() < 0.2:
                noise_level = 0.5
                noise = torch.randn_like(image) * noise_level
                image = image + noise
                logger.info("   🌪️ Input image is distorted/noisy!")
        else:
            # ダミーデータ
            image = torch.randn(1, 1, 28, 28)
            label = torch.tensor([0])
            
        return image.to(self.device), label.item(), noise_level

    def run_life_cycle(self, max_steps: int = 15):
        logger.info("🎬 Starting Visual Life Cycle...")
        
        try:
            for _ in range(max_steps):
                self.steps += 1
                print(f"\n--- Step {self.steps} ---")
                
                # 1. 知覚 (Perception)
                image, label, noise = self.get_visual_input()
                
                # 2. 思考 (Thinking)
                start_time = time.time()
                result = self.brain(image, noise_level=noise)
                latency = (time.time() - start_time) * 1000
                
                prediction = torch.argmax(result["logits"], dim=-1).item()
                system_used = result["system"]
                
                # 3. フィードバック
                is_correct = (prediction == label)
                result_str = "✅ Correct" if is_correct else f"❌ Wrong (Ans:{label})"
                
                logger.info(f"   👁️ Saw Digit: {label} | Prediction: {prediction} ({result_str})")
                logger.info(f"   🧠 Processed by: {system_used}")
                logger.info(f"   ⚡ Latency: {latency:.2f} ms")
                
                # 4. 学習と記憶 (Learning & Memory)
                # 間違えた場合や、System 2を使った場合は印象に残るため記憶する
                if not is_correct or "System 2" in system_used:
                    logger.info("   📝 Notable event. Consolidating to Hippocampus...")
                    
                    # 視覚トークンを記憶として保存 (MPS対策でCPUへ)
                    visual_memory = result["visual_tokens"].cpu() # (1, 49)
                    text_memory = torch.tensor([label]).cpu()     # 正解ラベル
                    reward = -1.0 if not is_correct else 1.0
                    
                    self.sleep_system.store_experience(
                        image=visual_memory,
                        text=text_memory,
                        reward=reward
                    )
                    self.fatigue += 0.25
                else:
                    self.fatigue += 0.05
                    
                logger.info(f"   🔋 Fatigue: {self.fatigue:.2f}/1.0")
                
                # 5. 睡眠チェック
                if self.fatigue >= 1.0:
                    logger.info("💤 Visual Cortex exhausted. Sleeping...")
                    summary = self.sleep_system.perform_sleep_cycle(duration_cycles=2)
                    logger.info(f"   -> Sleep Summary: {summary}")
                    self.fatigue = 0.0
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user.")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    agent = Phase4VisualAgent()
    agent.run_life_cycle(max_steps=20)