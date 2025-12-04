# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# (更新: Consciousness-Modulated Learning 対応)
# Title: 強化学習エージェント (Reinforcement Learner Agent)
# Description:
# - BioSNNと報酬変調型STDPを用い、トップダウンの因果クレジット信号で学習が変調されるエージェント。
# - 修正: `learn`メソッドに`global_context`引数を追加し、Global Workspaceの文脈情報を
#   学習則（CausalTraceなど）へ渡すことで、文脈依存の可塑性変調を実現。

import torch
from typing import Dict, Any, List, Optional

from snn_research.models.bio.simple_network import BioSNN
from snn_research.learning_rules.base_rule import BioLearningRule
from snn_research.communication import SpikeEncoderDecoder

class ReinforcementLearnerAgent:
    """
    BioSNNと報酬変調型STDPを用い、トップダウンの因果クレジット信号で学習が変調される強化学習エージェント。
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        device: str,
        synaptic_rule: BioLearningRule, # 外部から注入
        homeostatic_rule: Optional[BioLearningRule] = None # 外部から注入
    ):
        self.device = device
        
        hidden_size = (input_size + output_size) * 2
        layer_sizes = [input_size, hidden_size, output_size]
        
        self.model = BioSNN(
            layer_sizes=layer_sizes,
            neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0},
            synaptic_rule=synaptic_rule,
            homeostatic_rule=homeostatic_rule
        ).to(device)

        self.encoder = SpikeEncoderDecoder(num_neurons=input_size, time_steps=1)
        self.experience_buffer: List[List[torch.Tensor]] = []

    def get_action(self, state: torch.Tensor) -> int:
        """
        現在の状態から、モデルの推論によって単一の行動インデックスを決定する。
        """
        self.model.eval()
        with torch.no_grad():
            input_spikes = (torch.rand_like(state) < (state * 0.5 + 0.5)).float()
            output_spikes, hidden_spikes_history = self.model(input_spikes)
            self.experience_buffer.append([input_spikes] + hidden_spikes_history)
            action = torch.argmax(output_spikes).item()
            return int(action)

    def learn(self, reward: float, causal_credit: float = 0.0, global_context: Optional[Dict[str, Any]] = None):
        """
        受け取った報酬信号、因果的クレジット信号、およびグローバル文脈を用いて、モデルの重みを更新する。
        
        Args:
            reward (float): 環境からの報酬。
            causal_credit (float): 因果推論エンジンからのクレジット信号。
            global_context (Optional[Dict[str, Any]]): Global Workspaceからの意識内容（感情、注意など）。
        """
        if not self.experience_buffer:
            return

        self.model.train()
        
        # 因果的クレジット信号が与えられた場合、それを優先し、学習を増幅させる
        if causal_credit > 0:
            # クレジット信号は通常の報酬よりも強力な学習トリガーとする
            final_reward_signal = reward + causal_credit * 10.0 
            print(f"🧠 シナプス学習増強！ (Causal Credit: {causal_credit})")
        else:
            final_reward_signal = reward
            
        optional_params: Dict[str, Any] = {"reward": final_reward_signal}
        
        # 文脈情報を追加 (学習則側でこれを利用して可塑性を変調する)
        if global_context:
            optional_params["global_workspace_context"] = global_context
            # print(f"  - Context-modulated learning active: {global_context.get('type', 'unknown')}")

        for step_spikes in self.experience_buffer:
            self.model.update_weights(
                all_layer_spikes=step_spikes,
                optional_params=optional_params
            )
        
        # エピソードが終了、または強力な学習イベントが発生したらバッファをクリア
        if reward != -0.05 or causal_credit > 0:
            self.experience_buffer = []