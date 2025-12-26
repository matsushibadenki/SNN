# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# Title: 強化学習エージェント (Exploration-Aware Spike Recording Fix)
# Description:
#   強化学習において、ランダム探索(randint)で選択された行動も学習できるよう、
#   Experience Bufferに記録する出力スパイクを「選択された行動」で上書きする修正を追加。
#   閾値を下げて初期発火を促進。

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, cast

from snn_research.models.bio.simple_network import BioSNN
from snn_research.learning_rules.base_rule import BioLearningRule

class ReinforcementLearnerAgent:
    """
    BioSNNを用いた強化学習エージェント。
    不確実性駆動型学習と GRPO アルゴリズムをサポート。
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        device: str, 
        synaptic_rule: BioLearningRule, 
        homeostatic_rule: Optional[BioLearningRule] = None
    ):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        
        layer_sizes = [input_size, (input_size + output_size) * 2, output_size]
        
        # BioSNNの初期化
        # [修正] v_threshold を 1.0 -> 0.6 に下げて初期学習を促進
        self.model = BioSNN(
            layer_sizes=layer_sizes,
            neuron_params={'tau_mem': 10.0, 'v_threshold': 0.6, 'v_reset': 0.0, 'v_rest': 0.0},
            synaptic_rule=synaptic_rule,
            homeostatic_rule=homeostatic_rule,
            neuron_type="adaptive_lif"
        ).to(device)

        self.experience_buffer: List[List[torch.Tensor]] = []

    def get_action(self, state: torch.Tensor, record_experience: bool = True) -> int:
        """
        状態に基づきアクションを選択し、必要に応じてスパイク履歴を保存。
        ランダム探索時でも学習が進むよう、選択したアクションのスパイクを記録する。
        """
        self.model.eval()
        with torch.no_grad():
            # 入力エンコーディング
            if state.dtype == torch.float32 and state.max() <= 1.0 and state.min() >= 0.0:
                input_spikes = state
            else:
                input_spikes = (torch.rand_like(state) < (state * 0.5 + 0.5)).float()
            
            # (Batch dimension might be missing, ensure shape consistency if needed)
            if input_spikes.dim() == 1:
                input_spikes = input_spikes.unsqueeze(0) # (1, InputSize)

            output_spikes, hidden_history = self.model(input_spikes)
            
            # アクション決定ロジック
            is_random_action = False
            # 出力層の発火合計を確認
            if output_spikes.sum() > 0:
                action_idx = torch.argmax(output_spikes).item()
            else:
                # 発火なし -> ランダムアクション (探索)
                action_idx = torch.randint(0, self.output_size, (1,)).item()
                is_random_action = True

            if record_experience:
                # [重要修正]
                # モデルが発火しなかった(または異なる行動をとった)場合でも、
                # 「実際に選択した行動」を教師信号的に学習させるため、履歴上の出力スパイクを書き換える。
                # これにより STDP が Input -> ChosenAction の結合を強化できる。
                
                # hidden_history は各層の出力リストを想定 (最後が出力層)
                # リストをコピーして修正
                recorded_history = [h.clone() for h in hidden_history]
                
                # 出力層のスパイクを強制的に設定
                # output_spikes の形状は (Batch, OutputSize)
                target_spike = torch.zeros_like(output_spikes)
                target_spike[0, int(action_idx)] = 1.0
                
                # BioSNNの実装依存だが、通常 hidden_history の最後が出力層
                if len(recorded_history) > 0:
                    recorded_history[-1] = target_spike
                
                # バッファには [入力, 隠れ層..., 出力] の順で保存
                self.experience_buffer.append([input_spikes] + recorded_history)
                
            return int(action_idx)

    def learn(self, reward: float, causal_credit: float = 0.0, global_context: Optional[Dict[str, Any]] = None):
        """保存された経験に基づきオンライン学習を実行。"""
        if not self.experience_buffer:
            return

        self.model.train()
        optional_params: Dict[str, Any] = {
            "reward": reward + causal_credit * 10.0,
            "uncertainty": 0.5
        }
        if global_context:
            optional_params["global_workspace_context"] = global_context

        for step_spikes in self.experience_buffer:
            self.model.update_weights(
                all_layer_spikes=step_spikes,
                optional_params=optional_params
            )
        
        self.experience_buffer = []

    def sample_thought_trajectories(
        self, 
        initial_state: torch.Tensor, 
        env_step_func: Callable[[int], Tuple[torch.Tensor, float, bool, Any]], 
        num_samples: int = 4, 
        max_steps: int = 10
    ) -> List[Dict[str, Any]]:
        """GRPO等のための思考軌跡サンプリング。"""
        trajectories = []
        for _ in range(num_samples):
            current_state = initial_state.clone()
            trajectory: Dict[str, Any] = {
                'actions': [], 
                'rewards': [], 
                'spikes_history': [], 
                'total_reward': 0.0
            }
            
            # 各軌跡の開始時にバッファをクリア
            self.experience_buffer = []
            
            for _ in range(max_steps):
                action = self.get_action(current_state, record_experience=True)
                next_state, reward, done, _ = env_step_func(action)
                
                cast(List[int], trajectory['actions']).append(action)
                cast(List[float], trajectory['rewards']).append(reward)
                trajectory['total_reward'] = cast(float, trajectory['total_reward']) + reward
                
                current_state = next_state
                if done:
                    break
            
            trajectory['spikes_history'] = list(self.experience_buffer)
            trajectories.append(trajectory)
            
        self.experience_buffer = []
        return trajectories

    def learn_with_grpo(self, trajectories: List[Dict[str, Any]], baseline_reward: float = 0.0):
        """GRPO (Group Relative Policy Optimization) による方策改善。"""
        if not trajectories:
            return

        self.model.train()
        total_rewards = torch.tensor([t['total_reward'] for t in trajectories], dtype=torch.float32)
        
        # 報酬の正規化 (分散が0の場合は除算を避ける)
        mean_reward = total_rewards.mean()
        std_reward = total_rewards.std()
        if std_reward < 1e-6:
            std_reward = 1.0
            
        advantages = (total_rewards - mean_reward) / std_reward
        
        for i, trajectory in enumerate(trajectories):
            # アドバンテージを報酬として学習則に渡す
            advantage = advantages[i].item()
            
            # 大きすぎる勾配を防ぐためのクリッピング
            clipped_reward = float(np.clip(advantage, -2.0, 2.0))
            optional_params = {"reward": clipped_reward}
            
            for step_spikes in cast(List[List[torch.Tensor]], trajectory['spikes_history']):
                self.model.update_weights(
                    all_layer_spikes=step_spikes,
                    optional_params=optional_params
                )
