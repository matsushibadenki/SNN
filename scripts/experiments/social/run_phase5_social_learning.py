# ファイルパス: scripts/experiments/social/run_phase5_social_learning.py
# Title: Phase 5 Social Learning Experiment
# 修正内容: Mypyエラー修正 (型ヒントの追加、SNNCore引数修正、プロトコル対応)。

import torch
import torch.nn as nn
import logging
import random
import sys
import os
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from snn_research.core.snn_core import SNNCore
from snn_research.social.theory_of_mind import TheoryOfMindModule
# [Mypy Fix] 新しく定義したプロトコルクラスをインポート
from snn_research.social.emergent_language import EmergentLanguageProtocol

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialAgent:
    """社会学習を行うエージェント"""
    def __init__(self, agent_id: int, vocab_size: int = 50, hidden_dim: int = 128):
        self.id = agent_id
        self.device = "cpu"
        
        # [Mypy Fix] SNNCoreの初期化引数を修正
        config = {
            "architecture_type": "hybrid",
            "in_features": vocab_size,
            "hidden_features": hidden_dim,
            "out_features": vocab_size
        }
        self.brain = SNNCore(
            config=config,
            vocab_size=vocab_size
        ).to(self.device)
        
        # 社会的機能
        # [Mypy Fix] TheoryOfMindModuleの引数を修正 (agent_id削除)
        self.tom = TheoryOfMindModule(
            input_dim=vocab_size,
            hidden_dim=hidden_dim
        )
        self.language = EmergentLanguageProtocol(vocab_size=vocab_size)
        
        # 状態
        self.vocab_size = vocab_size
        self.trust_scores: Dict[int, float] = {}
        # 型ヒントを追加
        self.correct_history: List[bool] = [] 

    def listen(self, message: torch.Tensor, sender_id: int) -> torch.Tensor:
        """メッセージを受け取り、解釈して応答を生成"""
        # 1. 相手の意図を推定 (ToM)
        # [Mypy Fix] infer_intent -> forward (簡易対応) または observe_agentを使用
        # ここではメッセージを観測シーケンスとして扱う
        msg_seq = message.unsqueeze(0).unsqueeze(0) # (1, 1, vocab_size)
        intent = self.tom.forward(msg_seq)
        
        # 2. メッセージの解釈 (Brain)
        # 入力をSNNで処理
        brain_output = self.brain.forward(message.unsqueeze(0))
        
        # 3. 応答の生成
        response = torch.softmax(brain_output, dim=1)
        
        return response

    def learn(self, input_signal: torch.Tensor, feedback: float):
        """フィードバックに基づく学習"""
        # 簡易的な強化学習シグナル
        target = torch.argmax(input_signal).unsqueeze(0)
        # [Mypy Fix] SNNCoreに追加した update_plasticity を呼び出す
        self.brain.update_plasticity(
            x_input=input_signal.unsqueeze(0),
            target=target,
            learning_rate=0.01 * feedback
        )
        
        self.correct_history.append(feedback > 0)

def run_social_experiment():
    logger.info("👥 Starting Phase 5: Social Learning & Theory of Mind Experiment...")
    
    # 1. エージェントの初期化
    agents = [SocialAgent(i) for i in range(3)]
    logger.info(f"Initialized {len(agents)} social agents.")
    
    # 2. コミュニケーションループ (Naming Game)
    iterations = 50
    success_count = 0
    
    vocab_size = agents[0].vocab_size
    target_object = torch.zeros(vocab_size)
    target_object[random.randint(0, vocab_size-1)] = 1.0
    
    for i in range(iterations):
        # ランダムにペアを選択
        speaker_idx, listener_idx = random.sample(range(len(agents)), 2)
        speaker = agents[speaker_idx]
        listener = agents[listener_idx]
        
        # Speakerが発話
        message = speaker.listen(target_object, -1) # 自己対話でメッセージ生成
        
        # Listenerが解釈
        response = listener.listen(message.squeeze(0), speaker.id)
        
        # 合意判定 (Argmaxが一致するか)
        msg_token = torch.argmax(message).item()
        resp_token = torch.argmax(response).item()
        
        success = (msg_token == resp_token)
        reward = 1.0 if success else -0.1
        
        # 学習
        speaker.learn(target_object, reward)
        listener.learn(message.squeeze(0), reward)
        
        if success:
            success_count += 1
            
        if (i+1) % 10 == 0:
            logger.info(f"Iteration {i+1}: Agreement Rate {success_count/10:.2f}")
            success_count = 0
            
    logger.info("✅ Social learning experiment completed.")

if __name__ == "__main__":
    run_social_experiment()