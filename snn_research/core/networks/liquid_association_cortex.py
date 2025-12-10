# snn_research/core/networks/liquid_association_cortex.py
# Title: Liquid Association Cortex (Multi-modal Reservoir)
# Description: 
#   視覚、聴覚、体性感覚など異なるモダリティのスパイク入力を
#   単一のリカレントSNN（リザーバ）に統合し、連合記憶と時空間パターンを形成する。

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class LiquidAssociationCortex(nn.Module):
    def __init__(
        self,
        num_visual_inputs: int,
        num_audio_inputs: int,
        num_somato_inputs: int,
        reservoir_size: int = 1000,
        sparsity: float = 0.1
    ):
        super().__init__()
        
        # 1. 感覚ごとの入力ポート (Input Synapses)
        # ランダムな重みでリザーバ層へ投影
        self.visual_proj = nn.Linear(num_visual_inputs, reservoir_size, bias=False)
        self.audio_proj = nn.Linear(num_audio_inputs, reservoir_size, bias=False)
        self.somato_proj = nn.Linear(num_somato_inputs, reservoir_size, bias=False)
        
        # 2. リザーバ層 (Recurrent Synapses)
        # 固定重み (LSM) または STDP学習対象
        self.recurrent_weights = nn.Linear(reservoir_size, reservoir_size, bias=False)
        
        # スパース初期化 (脳のような疎結合を作る)
        self._init_sparse_weights(self.recurrent_weights, sparsity)
        
        # 3. ニューロンモデル (LIF)
        self.tau = 2.0
        self.threshold = 1.0
        
        # 状態保持
        self.mem = None
        self.spike = None

    def _init_sparse_weights(self, layer, sparsity):
        nn.init.kaiming_uniform_(layer.weight, a=0.1)
        # マスクをかけて結合を間引く
        mask = (torch.rand_like(layer.weight) < sparsity).float()
        layer.weight.data *= mask

    def forward(
        self, 
        visual_spikes: Optional[torch.Tensor] = None, 
        audio_spikes: Optional[torch.Tensor] = None,
        somato_spikes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            visual_spikes: (Batch, Inputs) - t時点のスパイク
            ...
        Returns:
            reservoir_spikes: (Batch, ReservoirSize) - 統合されたスパイク活動
        """
        batch_size = 0
        device = self.recurrent_weights.weight.device
        
        # 入力電流の合算 (Current Integration)
        current_input = 0.0
        
        if visual_spikes is not None:
            current_input += self.visual_proj(visual_spikes)
            batch_size = visual_spikes.size(0)
            
        if audio_spikes is not None:
            current_input += self.audio_proj(audio_spikes)
            batch_size = audio_spikes.size(0)
            
        if somato_spikes is not None:
            current_input += self.somato_proj(somato_spikes)
            batch_size = somato_spikes.size(0)
            
        # 状態初期化
        if self.mem is None or self.mem.size(0) != batch_size:
            self.mem = torch.zeros(batch_size, self.recurrent_weights.out_features).to(device)
            self.spike = torch.zeros(batch_size, self.recurrent_weights.out_features).to(device)

        # リカレント入力 (過去の自分からの影響)
        recurrent_input = self.recurrent_weights(self.spike)
        
        # 膜電位更新 (LIF Dynamics)
        # 入力 = 外部刺激 + 内部反響
        total_input = current_input + recurrent_input
        
        decay = torch.exp(torch.tensor(-1.0 / self.tau))
        self.mem = self.mem * decay + total_input * (1 - decay)
        
        # 発火判定
        new_spike = (self.mem >= self.threshold).float()
        self.mem = self.mem * (1.0 - new_spike) # リセット
        
        self.spike = new_spike
        return self.spike

    def reset_state(self):
        self.mem = None
        self.spike = None