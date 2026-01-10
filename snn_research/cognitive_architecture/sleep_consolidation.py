# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# 日本語タイトル: Sleep Consolidator (Hippocampal Replay System) v2.0
# 目的・内容:
#   ROADMAP Phase 2.1 "Sleep Consolidation" 対応。
#   覚醒中に蓄積したエピソード記憶（短期記憶）を、睡眠中にリプレイ学習（夢）させることで
#   長期記憶（ニューラルネットワークの重み）に定着させる。
#   優先度付き経験再生 (Prioritized Experience Replay) の簡易版として機能する。

import torch
import torch.nn as nn
import logging
import random
from typing import Dict
from collections import deque

logger = logging.getLogger(__name__)

class Episode:
    """
    1ステップの経験データ構造
    """
    def __init__(self, state: torch.Tensor, text: torch.Tensor, reward: float):
        self.state = state.cpu().detach() # GPUメモリ節約のためCPUへ退避
        self.text = text.cpu().detach()
        self.reward = reward

class SleepConsolidator:
    """
    睡眠による記憶固定システム。
    海馬(Hippocampus)の役割を模倣し、重要な記憶を大脳皮質(Cortex/Model)へ転送する。
    """
    
    def __init__(
        self, 
        agent: nn.Module, 
        optimizer: torch.optim.Optimizer,
        buffer_size: int = 1000,
        batch_size: int = 4,
        device: str = "cpu"
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        
        # 短期記憶バッファ (Hippocampus buffer)
        self.memory_buffer: deque = deque(maxlen=buffer_size)
        
        logger.info(f"💤 Sleep Consolidator initialized. Buffer size: {buffer_size}")

    def store_experience(self, image: torch.Tensor, text: torch.Tensor, reward: float):
        """
        覚醒中の経験をバッファに保存する。
        """
        episode = Episode(image, text, reward)
        self.memory_buffer.append(episode)

    def sleep(self, cycles: int = 5) -> Dict[str, float]:
        """
        睡眠モードを実行する。
        高報酬（驚きが大きかった）エピソードを優先的にサンプリングしてリプレイ学習を行う。
        """
        if len(self.memory_buffer) < self.batch_size:
            logger.warning("Not enough memories to sleep. Skipping.")
            return {"sleep_loss": 0.0}
            
        logger.info(f"🌙 Entering Sleep Mode... (Cycles: {cycles}, Memories: {len(self.memory_buffer)})")
        self.agent.train()
        self.agent.to(self.device)
        
        total_loss = 0.0
        
        # 記憶の優先順位付け (Rewardベース)
        # 報酬が高い順にソート（簡易的なPrioritized Replay）
        sorted_memories = sorted(self.memory_buffer, key=lambda x: x.reward, reverse=True)
        
        # 上位50%から重点的にサンプリング、下位からも少し混ぜる（多様性維持）
        top_tier = sorted_memories[:len(sorted_memories)//2]
        bottom_tier = sorted_memories[len(sorted_memories)//2:]
        
        for cycle in range(cycles):
            batch_episodes = []
            
            # Batch creation
            for _ in range(self.batch_size):
                if random.random() < 0.7 and top_tier: # 70% chance from top tier
                    batch_episodes.append(random.choice(top_tier))
                elif bottom_tier:
                    batch_episodes.append(random.choice(bottom_tier))
                else:
                    batch_episodes.append(random.choice(sorted_memories))
            
            # Tensor collation
            images = torch.cat([e.state for e in batch_episodes]).to(self.device)
            texts = torch.cat([e.text for e in batch_episodes]).to(self.device)
            
            # Replay Training (Dreaming)
            # ここでは「自己教師あり」的に、保存された画像とテキストの整合性を高める学習を行う
            # ※本来はNext Token Predictionの正解ラベルなども保存すべきだが、
            #   ここではVLMのAlignment Lossを最小化することで「概念の定着」を図る
            
            self.optimizer.zero_grad()
            
            outputs = self.agent(images, texts)
            
            # 睡眠学習の損失: 
            # 1. Alignment Loss (画像とテキストの結びつき強化)
            # 2. LogitsのEntropy抑制 (確信度を高める) などを入れても良いがシンプルに。
            loss = outputs["alignment_loss"]
            
            # 補足: もし生成タスクなら criterion_gen(outputs["logits"], targets) も必要
            # 今回は「概念結合」にフォーカスする
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Dream Visualization (Log first item of last cycle)
            if cycle == cycles - 1:
                logger.info(f"   💭 Dreaming... (Replaying memory with reward {batch_episodes[0].reward:.4f})")
                
        avg_loss = total_loss / cycles
        logger.info(f"🌅 Waking up. Sleep consolidation complete. Avg Loss: {avg_loss:.4f}")
        
        # 睡眠後はバッファをクリア（または減衰）させるのが一般的だが、
        # ここでは「重要な記憶は長期化された」としてクリアする設定にする
        self.memory_buffer.clear()
        
        return {"sleep_loss": avg_loss}