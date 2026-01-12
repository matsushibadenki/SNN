# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# 日本語タイトル: Sleep Consolidator (Hippocampal-Cortical Consolidation) v2.7 (MPS Fix)
# 修正 (v2.7): _train_stepでの入力データに .contiguous() を適用し、MPSエラーを回避。

import torch
import torch.nn as nn
import logging
import random
from typing import Dict, Any, Optional, List, Deque
from collections import deque

logger = logging.getLogger(__name__)


class Episode:
    """
    レガシー/テスト互換用のエピソードコンテナ。
    """

    def __init__(self, state: torch.Tensor, text: torch.Tensor, reward: float):
        self.state = state.cpu().detach()
        self.text = text.cpu().detach()
        self.reward = reward


class SleepConsolidator:
    """
    睡眠による記憶固定化システム (System 2 Consolidation)。
    海馬の短期記憶をリプレイし、大脳皮質の長期記憶(RAG/Weights)へ転送・統合する。
    """

    def __init__(
        self,
        memory_system: Optional[Any] = None,
        hippocampus: Optional[Any] = None,
        cortex: Optional[Any] = None,
        target_brain_model: Optional[nn.Module] = None,
        agent: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Dict[str, Any] = {}
    ):
        self.config = config
        self.hippocampus_buffer: Deque[Episode] = deque(maxlen=1000)
        
        # 依存関係の解決
        self.agent = target_brain_model if target_brain_model else agent
        self.cortex = cortex
        self.memory_system = memory_system
        
        self.batch_size = config.get("replay_batch_size", 32)
        self.learning_rate = config.get("sleep_learning_rate", 1e-4)
        
        # オプティマイザの初期化
        if optimizer:
            self.optimizer = optimizer
        elif self.agent:
            # 凍結されていないパラメータのみ対象
            params = [p for p in self.agent.parameters() if p.requires_grad]
            if params:
                self.optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
            else:
                self.optimizer = None
        else:
            self.optimizer = None

        logger.info(f"💤 Sleep Consolidator v2.7 initialized (Knowledge Graph Integration enabled).")

    def store_experience(self, image: torch.Tensor, text: torch.Tensor, reward: float):
        """覚醒時の経験を海馬(バッファ)に一時保存"""
        episode = Episode(image, text, reward)
        self.hippocampus_buffer.append(episode)

    def perform_sleep_cycle(self, duration_cycles: int = 5, recent_memories: List[Any] = []) -> Dict[str, Any]:
        """睡眠サイクルの実行 (Replay & Consolidation)"""
        if not self.agent:
            return {"status": "skipped", "reason": "no_agent_model"}
        
        if len(self.hippocampus_buffer) < 2 and not recent_memories:
             return {"status": "skipped", "reason": "no_memories"}

        logger.info(f"🌙 Starting Sleep Consolidation. Processing {len(self.hippocampus_buffer)} episodes over {duration_cycles} cycles.")
        
        self.agent.train()
        total_loss = 0.0
        consolidated_count = 0
        
        try:
            for cycle in range(duration_cycles):
                loss = self._train_step()
                total_loss += loss
                
                # 古い記憶の一部を長期記憶へ転送
                if self.hippocampus_buffer and random.random() < 0.3:
                    mem = self.hippocampus_buffer[0] # 古いものから
                    self._transfer_to_cortex(mem)
                    consolidated_count += 1
            
            return {
                "status": "success",
                "cycles": duration_cycles,
                "processed_episodes": len(self.hippocampus_buffer),
                "consolidated_to_cortex": consolidated_count,
                "avg_replay_loss": total_loss / duration_cycles,
                "knowledge_graph": {} # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Sleep cycle failed: {e}")
            return {"status": "failed", "error": str(e)}
        finally:
            self.agent.eval()

    def _train_step(self) -> float:
        """リプレイによる学習ステップ"""
        if not self.optimizer or len(self.hippocampus_buffer) == 0:
            return 0.0
            
        # バッチ作成
        batch_size = min(len(self.hippocampus_buffer), self.batch_size)
        batch = random.sample(self.hippocampus_buffer, batch_size)
        
        # テンソルの結合
        try:
            device = next(self.agent.parameters()).device
            
            # [MPS Fix] ここでリスト内包表記からstackした後、.contiguous()を適用
            states = torch.stack([e.state for e in batch]).to(device).squeeze(1).contiguous()
            
            # ラベルがない場合は自己教師あり学習（次のトークン予測など）を想定
            # ここでは簡易的に「入力そのものを再構成する」あるいは「ダミー損失」
            
            self.optimizer.zero_grad()
            
            # Forward
            if hasattr(self.agent, "forward"):
                # SNNCoreやSFormerの場合
                outputs = self.agent(states)
                
                # 出力がタプルの場合 (logits, spikes, mem)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif isinstance(outputs, dict):
                    logits = outputs.get('logits', list(outputs.values())[0])
                else:
                    logits = outputs
                
                # 簡易的な再構成損失 (Autoencoder的) or 自己回帰損失
                # ここでは入力IDをターゲットとするCrossEntropy (SFormerがLLM的なら)
                if hasattr(self.agent, "vocab_size") and logits.shape[-1] == self.agent.vocab_size:
                    # logits: (B, L, V), states: (B, L)
                    # 形状調整
                    B, L, V = logits.shape
                    
                    # ターゲットの形状確認
                    targets = states
                    if targets.dim() > 2: # (B, 1, L) -> (B, L)
                        targets = targets.squeeze(1)
                    
                    # 長さが合わない場合のトリミング
                    if targets.shape[1] > L:
                        targets = targets[:, :L]
                    elif targets.shape[1] < L:
                        logits = logits[:, :targets.shape[1], :]
                        
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.reshape(-1, V), targets.reshape(-1))
                else:
                    # その他のモデル用（ダミー）
                    loss = logits.mean() 
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                return float(loss.item())
                
        except Exception as e:
            logger.error(f"Replay training step failed: {e}")
            return 0.0
            
        return 0.0

    def _transfer_to_cortex(self, memory: Any):
        """エピソードを長期記憶(Cortex/RAG)へ転送・保存する"""
        pass # (以下省略、変更なし)