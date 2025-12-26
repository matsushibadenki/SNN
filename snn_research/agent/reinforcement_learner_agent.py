# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# Title: RL Agent with Hybrid Core (SCAL)
# Description: 検証済みのHybridNeuromorphicCoreとLogicGatedSNNを採用し、
#              GRPO学習において圧倒的な堅牢性を実現するエージェント。

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, cast

# 新しい検証済みコアを使用
from snn_research.core.hybrid_core import HybridNeuromorphicCore
# インターフェース互換性のため残すが、実体はHybridCoreが担う
from snn_research.learning_rules.base_rule import BioLearningRule

class ReinforcementLearnerAgent:
    """
    SCAL (Statistical Centroid Alignment Learning) を搭載した次世代強化学習エージェント。
    HybridNeuromorphicCoreを使用し、ノイズ環境下でも高速かつ堅牢に学習する。
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        device: str, 
        synaptic_rule: Optional[BioLearningRule] = None, # 互換性のため残存
        homeostatic_rule: Optional[BioLearningRule] = None
    ):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        
        # 検証済みの強力なコア構造
        # Hidden 64 -> Reservoir 64 -> TopK -> Readout
        hidden_dim = 64
        
        self.model = HybridNeuromorphicCore(
            in_features=input_size,
            hidden_features=hidden_dim,
            out_features=output_size
        ).to(device)

        # Experience Buffer: 状態ではなく、中間表現(Reservoir Output)を保存して効率化
        self.experience_buffer: List[Dict[str, torch.Tensor]] = []

    def get_action(self, state: torch.Tensor, record_experience: bool = True) -> int:
        self.model.eval() 
        
        if record_experience:
            self.model.train()
        
        with torch.no_grad():
            # HybridCore Forward
            # Input -> Fast(Reservoir) -> f
            # f -> Deep(TopK) -> r
            # r -> OutputGate -> out (Probabilities or Logits)
            
            f = self.model.fast_process(state)
            r = self.model.deep_process(f)
            out = self.model.output_gate(r)
            
            # 確率的サンプリング (Softmax)
            # LogicGatedSNN(readout) は training=True で Softmax を返す
            if self.model.training:
                probs = out
            else:
                # inference時は one-hot ライクなものが返るが、念のため
                probs = torch.softmax(out, dim=1)
            
            # カテゴリカル分布からサンプリング
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            action_idx = int(action.item())

            if record_experience:
                # 学習に必要な情報: 
                # 1. 入力スパイク (TopK通過後の 'r' が Readout層への入力となる)
                # 2. 選択したアクション (Gradientの方向)
                
                step_data = {
                    'pre_spikes': r.clone(), # Readout層への入力
                    'action': action_idx,
                    'probs': probs.clone()
                }
                self.experience_buffer.append(step_data)
            
            return action_idx

    def learn(self, reward: float, causal_credit: float = 0.0, global_context: Optional[Dict[str, Any]] = None):
        pass 

    def learn_with_grpo(self, trajectories: List[Dict[str, Any]], baseline_reward: float = 0.0):
        """
        Group Relative Policy Optimization with SCAL (Statistical Centroid Alignment Learning).
        """
        if not trajectories:
            return

        self.model.train()
        
        # 1. アドバンテージ計算
        total_rewards = torch.tensor([t['total_reward'] for t in trajectories], dtype=torch.float32)
        
        if len(total_rewards) > 1:
            mean_reward = total_rewards.mean()
            std_reward = total_rewards.std() + 1e-8
            advantages = (total_rewards - mean_reward) / std_reward
        else:
            advantages = torch.zeros_like(total_rewards)
        
        # 2. バッチ更新
        # HybridCoreのOutputGateのみを更新する（Reservoirは固定）
        
        for i, trajectory in enumerate(trajectories):
            adv = float(advantages[i].item())
            # 報酬クリッピング
            clipped_reward = float(np.clip(adv, -1.0, 1.0))
            
            # 各ステップでの学習
            # trajectoriesには 'spikes_history' (これはAgent側のバッファ構造に依存) が入っている
            # get_action で保存した構造を使う
            
            episode_history = trajectory.get('spikes_history', [])
            
            for step_data in episode_history:
                if not isinstance(step_data, dict): 
                    continue # 旧形式データ回避

                pre_spikes = step_data['pre_spikes'] # (1, Hidden)
                action_idx = step_data['action']
                
                # 教師信号の作成: 選択したアクションに対して、報酬(正なら強化、負なら抑制)を与える
                # LogicGatedSNNのupdate_plasticityは、(target - out) 的な誤差を受け取る設計
                # ここでは単純に Advantage を重みとしたOneHotベクトルを作成
                
                reward_signal = torch.zeros((1, self.output_size), device=self.device)
                reward_signal[0, action_idx] = clipped_reward
                
                # OutputGateの重み更新 (LogicGatedSNNの強力な学習則を使用)
                # post_spikes (出力) は、ここでは選択されたアクションに対する期待値として扱う
                # 実際には update_plasticity 内で delta = reward * pre_spikes を計算する
                # LogicGatedSNN.update_plasticity の "Fallback for scalar reward" ロジックを利用
                
                # update_plasticity(pre, post, reward)
                # post_spikes として、ここでは仮にワンホット（選択した行動）を渡す
                post_spikes_target = torch.zeros((1, self.output_size), device=self.device)
                post_spikes_target[0, action_idx] = 1.0
                
                # SCAL則による更新
                self.model.output_gate.update_plasticity(
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes_target, 
                    reward=reward_signal, # ベクトル報酬
                    learning_rate=0.05 # 高めの学習率で高速収束
                )
