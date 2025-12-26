# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# Title: 強化学習エージェント (Exploration-Aware Learning Fix)
# Description:
#   BioSNNの修正に伴い、学習ロジックを強化。
#   ランダム探索時も、選択した行動に対応する出力ニューロンが発火したとみなして学習させる。

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, cast

# 修正された BioSNN をインポート
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
        
        # ネットワーク構成: 入力 -> 隠れ層 -> 出力
        # 隠れ層を少し広めにとって表現力を上げる
        layer_sizes = [input_size, (input_size + output_size) * 4, output_size]
        
        # [チューニング] 閾値を低めに設定して発火しやすくする (1.0 -> 0.5)
        self.model = BioSNN(
            layer_sizes=layer_sizes,
            neuron_params={'tau_mem': 20.0, 'v_threshold': 0.5, 'v_reset': 0.0, 'dt': 1.0},
            synaptic_rule=synaptic_rule,
            homeostatic_rule=homeostatic_rule,
            neuron_type="adaptive_lif" # BioSNN側ではkwargsで受け取るが、内部実装はLIF
        ).to(device)

        self.experience_buffer: List[List[torch.Tensor]] = []

    def get_action(self, state: torch.Tensor, record_experience: bool = True) -> int:
        """
        状態に基づきアクションを選択。
        探索（ランダム行動）時でも学習が進むよう、選択した行動のスパイクを記録する。
        """
        self.model.eval()
        with torch.no_grad():
            # 入力エンコーディング (Float -> Spike or Pass-through)
            if state.dtype == torch.float32 and state.max() <= 1.0 and state.min() >= 0.0:
                input_spikes = state
            else:
                input_spikes = (torch.rand_like(state) < (state * 0.5 + 0.5)).float()

            # バッチ次元の確保 (1, InputSize)
            if input_spikes.dim() == 1:
                input_spikes = input_spikes.unsqueeze(0)

            # 順伝播
            output_spikes, hidden_history = self.model(input_spikes)
            
            # アクション決定
            action_idx: int
            if output_spikes.sum() > 0:
                # 最も活性化したニューロンを選択
                # (Batch=1 なので [0] を参照)
                # バイナリスパイクの場合、重複があれば最初のindexになるが、電圧などを参照したほうが精度は良い。
                # ここでは簡易的にスパイクの有無で判断。
                action_idx = int(torch.argmax(output_spikes).item())
            else:
                # 発火なし -> ランダム探索
                action_idx = int(torch.randint(0, self.output_size, (1,)).item())

            if record_experience:
                # [重要修正] 強制スパイク記録 (Hindsight Experience Replay的な発想)
                # モデルが発火しなかった、またはランダム行動をとった場合、
                # 「実際に選択した行動」に対応するニューロンが発火したことにして履歴に残す。
                # これにより、偶然正解した行動に対する入力パターンを学習できる。
                
                final_history = [h.clone() for h in hidden_history]
                
                # 出力層の履歴を上書き
                target_output = torch.zeros_like(output_spikes)
                target_output[0, action_idx] = 1.0
                
                # historyの最後が出力層
                if len(final_history) > 0:
                    final_history[-1] = target_output
                
                # バッファには [Input, Hidden..., Output] として保存される
                # hidden_history は [Input, Hidden..., Output] を既に含んでいる仕様(BioSNN実装依存)
                # BioSNNの実装では spikes_history = [x, spikes_1, ..., spikes_out] なのでそのまま使う
                
                self.experience_buffer.append(final_history)
            
            return action_idx

    def learn(self, reward: float, causal_credit: float = 0.0, global_context: Optional[Dict[str, Any]] = None):
        """学習実行"""
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
        """GRPO用サンプリング"""
        trajectories = []
        for _ in range(num_samples):
            current_state = initial_state.clone()
            
            # リセット
            if hasattr(self.model, 'reset_state'):
                self.model.reset_state(batch_size=1, device=self.model.weights[0].device)
                
            trajectory: Dict[str, Any] = {
                'actions': [], 'rewards': [], 'spikes_history': [], 'total_reward': 0.0
            }
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
        """GRPO学習"""
        if not trajectories:
            return

        self.model.train()
        total_rewards = torch.tensor([t['total_reward'] for t in trajectories], dtype=torch.float32)
        mean_reward = total_rewards.mean()
        std_reward = total_rewards.std() + 1e-8
        advantages = (total_rewards - mean_reward) / std_reward
        
        for i, trajectory in enumerate(trajectories):
            optional_params = {"reward": float(np.clip(advantages[i].item(), -2.0, 2.0))}
            
            for step_spikes in cast(List[List[torch.Tensor]], trajectory['spikes_history']):
                self.model.update_weights(
                    all_layer_spikes=step_spikes,
                    optional_params=optional_params
                )
