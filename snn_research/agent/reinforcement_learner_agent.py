# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# (更新: GRPO & Consciousness-Modulated Learning 対応)
# Title: 強化学習エージェント (Reinforcement Learner Agent with GRPO)
# Description:
# - Phase 8-3: GRPO (Group Relative Policy Optimization) を実装。
# - 同一の状態から複数の「思考の軌跡」を生成し、グループ平均に対する優位性(Advantage)
#   を用いてシナプス可塑性を変調する。
# - 従来の learn メソッド (単一エピソード学習) と GRPO メソッド (グループ学習) を併用可能。

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable

from snn_research.models.bio.simple_network import BioSNN
from snn_research.learning_rules.base_rule import BioLearningRule
from snn_research.communication import SpikeEncoderDecoder

class ReinforcementLearnerAgent:
    """
    BioSNNと報酬変調型STDPを用い、GRPOによる論理推論強化を行ったエージェント。
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
        
        hidden_size = (input_size + output_size) * 2
        layer_sizes = [input_size, hidden_size, output_size]
        
        self.model = BioSNN(
            layer_sizes=layer_sizes,
            neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0},
            synaptic_rule=synaptic_rule,
            homeostatic_rule=homeostatic_rule
        ).to(device)

        self.encoder = SpikeEncoderDecoder(num_neurons=input_size, time_steps=1)
        
        # 通常のRL用バッファ
        self.experience_buffer: List[List[torch.Tensor]] = []

    def get_action(self, state: torch.Tensor, record_experience: bool = True) -> int:
        """
        現在の状態から、モデルの推論によって単一の行動インデックスを決定する。
        """
        self.model.eval()
        with torch.no_grad():
            # 入力がバイナリでない場合は確率的にスパイク化、既にスパイクならそのまま
            if state.max() > 1.0 or state.dtype == torch.float32:
                 input_spikes = (torch.rand_like(state) < (state * 0.5 + 0.5)).float()
            else:
                 input_spikes = state

            output_spikes, hidden_spikes_history = self.model(input_spikes)
            
            if record_experience:
                self.experience_buffer.append([input_spikes] + hidden_spikes_history)
            
            # 行動選択: 出力スパイクの中で最も発火したニューロン、なければランダム
            if output_spikes.sum() > 0:
                action = torch.argmax(output_spikes).item()
            else:
                action = torch.randint(0, self.output_size, (1,)).item()
                
            return int(action)

    def learn(self, reward: float, causal_credit: float = 0.0, global_context: Optional[Dict[str, Any]] = None):
        """
        従来の単一エピソード学習。
        """
        if not self.experience_buffer:
            return

        self.model.train()
        
        if causal_credit > 0:
            final_reward_signal = reward + causal_credit * 10.0 
        else:
            final_reward_signal = reward
            
        optional_params: Dict[str, Any] = {"reward": final_reward_signal}
        
        if global_context:
            optional_params["global_workspace_context"] = global_context

        for step_spikes in self.experience_buffer:
            self.model.update_weights(
                all_layer_spikes=step_spikes,
                optional_params=optional_params
            )
        
        # バッファクリア条件 (エピソード終了や強い学習時)
        # 簡易化のため毎回クリアとする
        self.experience_buffer = []

    # --- GRPO Implementation (Phase 8-3) ---

    def sample_thought_trajectories(
        self, 
        initial_state: torch.Tensor, 
        env_step_func: Callable[[int], Tuple[torch.Tensor, float, bool, Any]],
        num_samples: int = 4, 
        max_steps: int = 10
    ) -> List[Dict[str, Any]]:
        """
        GRPO用: 同一の初期状態から複数の「思考の軌跡(Trajectories)」を生成する。
        SNNの確率的挙動を利用して、異なる解法を探索させる。
        
        Args:
            initial_state: 初期状態
            env_step_func: 環境のstep関数ラッパー (action -> next_state, reward, done, info)
            num_samples: 生成する軌跡の数 (Group Size)
            max_steps: 1軌跡あたりの最大ステップ数
            
        Returns:
            List of trajectories (each containing actions, rewards, spike_history)
        """
        trajectories = []
        
        for _ in range(num_samples):
            # エージェントの状態をリセット (短期記憶のクリアなど)
            # ※ BioSNNの膜電位リセットなどは forward 時に行われる前提、またはここで明示的に reset メソッドを呼ぶ
            # self.model.reset_state() # もしモデルがステートフルなら
            
            current_state = initial_state.clone()
            trajectory = {
                'actions': [],
                'rewards': [],
                'spikes_history': [], # 学習用: 各ステップの全層スパイク活動
                'total_reward': 0.0
            }
            
            # 通常バッファを退避・クリアして、この軌跡専用の履歴をとる
            self.experience_buffer = []
            
            for _ in range(max_steps):
                action = self.get_action(current_state, record_experience=True)
                next_state, reward, done, _ = env_step_func(action)
                
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['total_reward'] += reward
                
                current_state = next_state
                if done:
                    break
            
            # 記録されたスパイク履歴を軌跡データに移動
            trajectory['spikes_history'] = list(self.experience_buffer)
            trajectories.append(trajectory)
            
        # バッファをクリーンアップ
        self.experience_buffer = []
        return trajectories

    def learn_with_grpo(self, trajectories: List[Dict[str, Any]], baseline_reward: float = 0.0):
        """
        GRPO (Group Relative Policy Optimization) による学習実行。
        
        グループ内の報酬の平均と分散を計算し、各軌跡の Advantage を算出。
        Advantage = (TrajectoryReward - GroupMean) / (GroupStd + epsilon)
        
        この Advantage を「変調信号」としてシナプス可塑性に適用する。
        """
        if not trajectories:
            return

        self.model.train()
        
        # 1. グループ統計量の計算
        total_rewards = torch.tensor([t['total_reward'] for t in trajectories], dtype=torch.float32)
        group_mean = total_rewards.mean()
        group_std = total_rewards.std() + 1e-8
        
        # ベースライン（過去の移動平均など）がある場合はそれも考慮可能だが、
        # GRPOの基本はグループ内相対評価。
        
        # 2. Advantage の計算
        # 平均より良い軌跡は正の学習信号、悪い軌跡は負の学習信号(忘却/抑制)を受ける
        advantages = (total_rewards - group_mean) / group_std
        
        print(f"🧠 GRPO Update: Group Mean Reward: {group_mean:.4f}, Std: {group_std:.4f}")
        
        # 3. 各軌跡に対する重み更新
        for i, trajectory in enumerate(trajectories):
            adv = advantages[i].item()
            
            # ノイズ等でAdvantageが極端にならないようクリッピング推奨
            adv = np.clip(adv, -2.0, 2.0)
            
            # スパイク履歴に基づいて学習
            # Advantage を reward 引数として渡すことで、LearningRule側で符号に応じた強化/抑制を行う
            optional_params = {"reward": adv}
            
            # トラジェクトリ内の全ステップに対して更新を適用
            # (より高度な実装では、ステップごとの減衰率(gamma)を考慮するが、ここではエピソード全体評価とする)
            for step_spikes in trajectory['spikes_history']:
                self.model.update_weights(
                    all_layer_spikes=step_spikes,
                    optional_params=optional_params
                )
