# ファイルパス: snn_research/models/bio/simple_network.py
# Title: 生物学的SNN (Implemented LIF & STDP)
# Description:
#   ダミー実装だったBioSNNを、実際に動作するLIF (Leaky Integrate-and-Fire) ネットワークに変更。
#   順伝播でスパイクを生成し、STDPルールに基づいて重みを更新する機能を追加。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, cast
import logging
from snn_research.core.base import BaseModel
from snn_research.learning_rules.base_rule import BioLearningRule

logger = logging.getLogger(__name__)

class BioSNN(BaseModel):
    """
    生物学的妥当性を備えた多層SNN。
    手動で重み行列(nn.Parameter)と膜電位を管理し、スパイク伝播を行う。
    """
    def __init__(
        self, 
        layer_sizes: List[int], 
        neuron_params: Dict[str, Any], 
        synaptic_rule: BioLearningRule, 
        homeostatic_rule: Optional[BioLearningRule] = None,
        sparsification_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.neuron_params = neuron_params
        
        # パラメータ設定
        self.tau_mem = neuron_params.get('tau_mem', 10.0)
        self.v_threshold = neuron_params.get('v_threshold', 1.0)
        self.v_reset = neuron_params.get('v_reset', 0.0)
        self.dt = neuron_params.get('dt', 1.0)
        
        # 重み行列の定義 (Input->Hidden, Hidden->Output...)
        self.weights = nn.ParameterList()
        self.synaptic_rules: List[BioLearningRule] = []
        
        # 内部状態（膜電位）の保持用
        self.mem_potentials: List[torch.Tensor] = []

        # 重みの初期化と学習ルールの複製
        import copy
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot Initialization
            w_init = torch.randn(layer_sizes[i], layer_sizes[i+1]) / (layer_sizes[i] ** 0.5)
            self.weights.append(nn.Parameter(w_init))
            self.synaptic_rules.append(copy.deepcopy(synaptic_rule))
            
        # 刈り込み設定
        config = sparsification_config or {}
        self.sparsification_enabled = config.get("enabled", False)

    def reset_state(self, batch_size: int, device: torch.device):
        """膜電位のリセット"""
        self.mem_potentials = []
        for size in self.layer_sizes[1:]: # 入力層以外
            self.mem_potentials.append(torch.zeros(batch_size, size, device=device))

    def apply_causal_pruning(self, layer_idx: int) -> None:
        """因果貢献度に基づく刈り込み"""
        rule = self.synaptic_rules[layer_idx]
        if hasattr(rule, 'get_causal_contribution'):
            contribution = cast(Any, rule).get_causal_contribution()
            if contribution is not None:
                # 下位10%を刈り込む
                threshold = torch.quantile(contribution.abs(), 0.1)
                mask = (contribution.abs() >= threshold).float()
                with torch.no_grad():
                    # contributionの形状によっては転置が必要な場合があるが、
                    # ここでは学習則の実装に依存。通常は重みと同じ形状を想定。
                    if contribution.shape == self.weights[layer_idx].shape:
                        self.weights[layer_idx].data.mul_(mask)
                    elif contribution.t().shape == self.weights[layer_idx].shape:
                         self.weights[layer_idx].data.mul_(mask.t())

    def update_weights(self, all_layer_spikes: List[torch.Tensor], optional_params: Optional[Dict[str, Any]] = None) -> None:
        """
        STDP学習則による重み更新。
        all_layer_spikes: [Input, Hidden1, ..., Output] のスパイク列リスト
        """
        uncertainty = (optional_params or {}).get("uncertainty", 1.0)
        
        # 各レイヤー間で重みを更新 (Input->Hidden, Hidden->Output)
        for i in range(len(self.weights)):
            pre_spikes = all_layer_spikes[i]
            post_spikes = all_layer_spikes[i+1]
            rule = self.synaptic_rules[i]
            
            # weights[i] は (Pre, Post) または (Post, Pre) の形状
            # ここでは (Pre, Post) を想定して matmul しているのでそれに合わせる
            # STDPルールは通常 (Post, Pre) の dw を返すことが多いが、ルール次第。
            # 今回の RewardModulatedSTDP は weights の形状に合わせてくれると仮定。
            
            dw, _ = rule.update(pre_spikes, post_spikes, self.weights[i], optional_params)
            
            with torch.no_grad():
                self.weights[i].add_(dw)
                # 重みのクリッピング (発散防止)
                self.weights[i].clamp_(-1.0, 1.0)
            
            # 刈り込み
            if self.sparsification_enabled and uncertainty < 0.3:
                self.apply_causal_pruning(i)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        順伝播計算 (LIFニューロン)
        Args:
            x: 入力スパイク (Batch, InputSize)
        Returns:
            output_spikes: 出力層のスパイク
            history: 全層のスパイク活動 [Input, Hidden..., Output]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 初回またはバッチサイズ変更時にリセット
        if not self.mem_potentials or self.mem_potentials[0].shape[0] != batch_size:
            self.reset_state(batch_size, device)
            
        spikes_history = [x]
        current_input = x
        
        for i, weight in enumerate(self.weights):
            # 電流入力 I = Input @ W
            # weight: (Pre, Post)
            current = torch.matmul(current_input, weight)
            
            # 膜電位更新: V(t) = V(t-1) * decay + I
            decay = 1.0 - (self.dt / self.tau_mem)
            self.mem_potentials[i] = self.mem_potentials[i] * decay + current
            
            # 発火判定
            spikes = (self.mem_potentials[i] >= self.v_threshold).float()
            
            # リセット (Soft Reset: 閾値を引く)
            self.mem_potentials[i] = self.mem_potentials[i] - (spikes * self.v_threshold)
            
            # 次の層への入力
            current_input = spikes
            spikes_history.append(spikes)
            
        return current_input, spikes_history
