# snn_research/core/networks/liquid_association_cortex.py
# Title: Liquid Association Cortex (Multi-modal Reservoir) - Type Safe
# Description: 
#   視覚、聴覚、体性感覚など異なるモダリティのスパイク入力を
#   単一のリカレントSNN（リザーバ）に統合する。mypyのエラーを修正した型安全版。

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, cast

class LiquidAssociationCortex(nn.Module):
    # 型ヒントをクラスレベルで明示
    mem: Optional[torch.Tensor]
    spike: Optional[torch.Tensor]

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
        self.visual_proj = nn.Linear(num_visual_inputs, reservoir_size, bias=False)
        self.audio_proj = nn.Linear(num_audio_inputs, reservoir_size, bias=False)
        self.somato_proj = nn.Linear(num_somato_inputs, reservoir_size, bias=False)
        
        # 2. リザーバ層 (Recurrent Synapses)
        self.recurrent_weights = nn.Linear(reservoir_size, reservoir_size, bias=False)
        
        # スパース初期化
        self._init_sparse_weights(self.recurrent_weights, sparsity)
        
        # 3. ニューロンパラメータ
        self.tau = 2.0
        self.threshold = 1.0
        
        # 状態保持 (初期値はNone)
        self.mem = None
        self.spike = None

    def _init_sparse_weights(self, layer: nn.Linear, sparsity: float) -> None:
        nn.init.kaiming_uniform_(layer.weight, a=0.1)
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
            visual_spikes: (Batch, Inputs)
        Returns:
            reservoir_spikes: (Batch, ReservoirSize)
        """
        batch_size = 0
        device = self.recurrent_weights.weight.device
        
        # 入力電流の合算
        current_input = torch.tensor(0.0, device=device)
        
        if visual_spikes is not None:
            current_input = current_input + self.visual_proj(visual_spikes)
            batch_size = visual_spikes.size(0)
            
        if audio_spikes is not None:
            current_input = current_input + self.audio_proj(audio_spikes)
            batch_size = audio_spikes.size(0)
            
        if somato_spikes is not None:
            current_input = current_input + self.somato_proj(somato_spikes)
            batch_size = somato_spikes.size(0)
            
        # 入力が全くない場合のガード
        if batch_size == 0:
            # 既存の状態があればそれを返す、なければ空のダミーを返す
            if self.spike is not None:
                return self.spike
            # 初期化前に入力なしで呼ばれた場合
            return torch.zeros(1, self.recurrent_weights.out_features, device=device)

        # 状態初期化またはサイズ不一致時のリセット
        # ローカル変数に代入して型を確定させる
        mem = self.mem
        spike = self.spike

        if mem is None or spike is None or mem.size(0) != batch_size:
            mem = torch.zeros(batch_size, self.recurrent_weights.out_features).to(device)
            spike = torch.zeros(batch_size, self.recurrent_weights.out_features).to(device)

        # リカレント入力
        recurrent_input = self.recurrent_weights(spike)
        
        # 膜電位更新
        total_input = current_input + recurrent_input
        
        decay = torch.exp(torch.tensor(-1.0 / self.tau, device=device))
        mem = mem * decay + total_input * (1 - decay)
        
        # 発火判定 (bool -> float キャストを明示)
        # mypyが (Tensor >= float) の戻り値を bool と誤認する場合があるため 1.0 を掛ける等で回避
        is_fire = (mem >= self.threshold)
        new_spike = is_fire.float()
        
        # リセット
        mem = mem * (1.0 - new_spike)
        
        # 状態の書き戻し
        self.mem = mem
        self.spike = new_spike
        
        return new_spike

    def reset_state(self) -> None:
        self.mem = None
        self.spike = None
