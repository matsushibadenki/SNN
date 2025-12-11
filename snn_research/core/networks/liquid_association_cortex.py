# ファイルパス: snn_research/core/networks/liquid_association_cortex.py
# 日本語タイトル: Liquid Association Cortex (Multi-modal Reservoir) [Extended]
# 目的・内容:
#   Phase 9-9: 視覚、聴覚、体性感覚、そして「言語」のスパイク入力を
#   単一のリカレントSNN（リザーバ）に統合する「液状連合野」。
#   Universal Spike Encoderからの出力を受け取り、モダリティを超えた連想記憶の基盤となる。

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, cast

class LiquidAssociationCortex(nn.Module):
    # 型ヒント
    mem: Optional[torch.Tensor]
    spike: Optional[torch.Tensor]

    def __init__(
        self,
        num_visual_inputs: int,
        num_audio_inputs: int,
        num_text_inputs: int,   # Added: 言語入力
        num_somato_inputs: int,
        reservoir_size: int = 1000,
        sparsity: float = 0.1,
        tau: float = 2.0,
        threshold: float = 1.0
    ):
        super().__init__()
        
        # 1. 感覚ごとの入力ポート (Input Synapses)
        # スパースな投影を行うことで、リザーバ内の「視覚領域」「聴覚領域」が緩やかに形成されることを期待
        self.visual_proj = nn.Linear(num_visual_inputs, reservoir_size, bias=False)
        self.audio_proj = nn.Linear(num_audio_inputs, reservoir_size, bias=False)
        self.text_proj = nn.Linear(num_text_inputs, reservoir_size, bias=False) # Added
        self.somato_proj = nn.Linear(num_somato_inputs, reservoir_size, bias=False)
        
        # 2. リザーバ層 (Recurrent Synapses)
        # ランダムかつスパースに結合されたリカレント層
        self.recurrent_weights = nn.Linear(reservoir_size, reservoir_size, bias=False)
        
        # スパース初期化 (Spectral Radius調整などは簡易的に済ます)
        self._init_sparse_weights(self.recurrent_weights, sparsity)
        
        # 3. ニューロンパラメータ
        self.tau = tau
        self.threshold = threshold
        
        # 状態保持
        self.mem = None
        self.spike = None

    def _init_sparse_weights(self, layer: nn.Linear, sparsity: float) -> None:
        """Kaiming初期化後にマスクを適用してスパース化"""
        nn.init.kaiming_uniform_(layer.weight, a=0.1)
        with torch.no_grad():
            mask = (torch.rand_like(layer.weight) < sparsity).float()
            layer.weight.data *= mask
            # リザーバの安定性のため、スペクトル半径を調整するのが一般的だが今回は省略

    def forward(
        self, 
        visual_spikes: Optional[torch.Tensor] = None, 
        audio_spikes: Optional[torch.Tensor] = None,
        text_spikes: Optional[torch.Tensor] = None, # Added
        somato_spikes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            *_spikes: (Batch, Inputs) - 各タイムステップのスパイク入力
        Returns:
            reservoir_spikes: (Batch, ReservoirSize) - リザーバ層のスパイク出力
        """
        device = self.recurrent_weights.weight.device
        batch_size = 0
        
        # 入力電流の合算 (Current Summation)
        current_input = torch.tensor(0.0, device=device)
        
        if visual_spikes is not None:
            current_input = current_input + self.visual_proj(visual_spikes)
            batch_size = visual_spikes.size(0)
            
        if audio_spikes is not None:
            current_input = current_input + self.audio_proj(audio_spikes)
            batch_size = max(batch_size, audio_spikes.size(0))
            
        if text_spikes is not None:
            current_input = current_input + self.text_proj(text_spikes)
            batch_size = max(batch_size, text_spikes.size(0))
            
        if somato_spikes is not None:
            current_input = current_input + self.somato_proj(somato_spikes)
            batch_size = max(batch_size, somato_spikes.size(0))
            
        # 入力が全くない場合のガード
        if batch_size == 0:
            if self.spike is not None:
                return self.spike
            return torch.zeros(1, self.recurrent_weights.out_features, device=device)

        # 状態初期化またはサイズ不一致時のリセット
        mem = self.mem
        spike = self.spike

        if mem is None or spike is None or mem.size(0) != batch_size:
            mem = torch.zeros(batch_size, self.recurrent_weights.out_features).to(device)
            spike = torch.zeros(batch_size, self.recurrent_weights.out_features).to(device)

        # リカレント入力: 前ステップのスパイク * 再帰重み
        recurrent_input = self.recurrent_weights(spike)
        
        # 膜電位更新 (LIF Dynamics)
        # V[t] = V[t-1] * decay + I_in + I_rec
        total_input = current_input + recurrent_input
        
        decay = torch.exp(torch.tensor(-1.0 / self.tau, device=device))
        mem = mem * decay + total_input * (1 - decay)
        
        # 発火判定
        is_fire = (mem >= self.threshold)
        new_spike = is_fire.float()
        
        # ソフトリセット (減算リセット)
        mem = mem - (new_spike * self.threshold)
        
        # 状態の書き戻し
        self.mem = mem
        self.spike = new_spike
        
        return new_spike

    def reset_state(self) -> None:
        self.mem = None
        self.spike = None
