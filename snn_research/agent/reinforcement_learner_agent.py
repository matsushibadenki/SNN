# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# Title: RL Agent with Hybrid Core (SCAL & Reset)
# Description: エピソード間の状態リセットを実装し、GRPOの安定性を確保。

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, cast

# 新しい検証済みコアを使用
from snn_research.core.hybrid_core import HybridNeuromorphicCore
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
        synaptic_rule: Optional[BioLearningRule] = None, 
        homeostatic_rule: Optional[BioLearningRule] = None
    ):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        
        # 検証済みの強力なコア構造
        hidden_dim = 64
        
        self.model = HybridNeuromorphicCore(
            in_features=input_size,
            hidden_features=hidden_dim,
            out_features=output_size
        ).to(device)

        # Experience Buffer
        self.experience_buffer: List[Dict[str, torch.Tensor]] = []

    def get_action(self, state: torch.Tensor, record_experience: bool = True) -> int:
        self.model.eval() 
        
        if record_experience:
            self.model.train()
        
        with torch.no_grad():
            f = self.model.fast_process(state)
            r = self.model.deep_process(f)
            out = self.model.output_gate(r)
            
            if self.model.training:
                # 探索時は少し温度を上げてランダム性を増やす (Temperature=1.2)
                probs = torch.softmax(out / 1.2, dim=1)
            else:
                probs = torch.softmax(out, dim=1)
            
            # カテゴリカル分布からサンプリング
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            action_idx = int(action.item())

            if record_experience:
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
        if not trajectories:
            return

        self.model.train()
        
        total_rewards = torch.tensor([t['total_reward'] for t in trajectories], dtype=torch.float32)
        
        if len(total_rewards) > 1:
            mean_reward = total_rewards.mean()
            std_reward = total_rewards.std() + 1e-8
            advantages = (total_rewards - mean_reward) / std_reward
        else:
            advantages = torch.zeros_like(total_rewards)
        
        for i, trajectory in enumerate(trajectories):
            # [修正] 新しいエピソードの学習前に状態をリセットし、干渉を防ぐ
            self.model.reset_state()
            
            adv = float(advantages[i].item())
            clipped_reward = float(np.clip(adv, -1.0, 1.0))
            
            episode_history = trajectory.get('spikes_history', [])
            
            for step_data in episode_history:
                if not isinstance(step_data, dict): 
                    continue 

                pre_spikes = step_data['pre_spikes'] 
                action_idx = step_data['action']
                
                # 教師信号（報酬ベクトル）の作成
                reward_signal = torch.zeros((1, self.output_size), device=self.device)
                reward_signal[0, action_idx] = clipped_reward
                
                # ダミーのTarget Spike (LogicGatedSNNのインターフェース合わせ)
                post_spikes_target = torch.zeros((1, self.output_size), device=self.device)
                post_spikes_target[0, action_idx] = 1.0
                
                # SCAL則による更新
                self.model.output_gate.update_plasticity(
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes_target, 
                    reward=reward_signal, 
                    learning_rate=0.05
                )
