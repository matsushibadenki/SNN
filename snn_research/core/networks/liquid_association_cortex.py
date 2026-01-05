# ファイルパス: snn_research/core/networks/liquid_association_cortex.py
# Title: Liquid Association Cortex (Multi-modal Reservoir) [Activity Boosted]
# Description:
#   Phase 9-10: デモでの発火不足を解消するための修正版。
#   - input_scale: 入力電流のスケーリング係数を追加。
#   - 重み初期化のゲインを強化。

import torch
import torch.nn as nn
from typing import Optional

from snn_research.learning_rules.base_rule import BioLearningRule


class LiquidAssociationCortex(nn.Module):
    # 型ヒント
    mem: Optional[torch.Tensor]
    spike: Optional[torch.Tensor]
    learning_rule: Optional[BioLearningRule]

    def __init__(
        self,
        num_visual_inputs: int,
        num_audio_inputs: int,
        num_text_inputs: int,
        num_somato_inputs: int,
        reservoir_size: int = 1000,
        sparsity: float = 0.1,
        tau: float = 2.0,
        threshold: float = 1.0,
        input_scale: float = 10.0,  # Added: 入力ゲイン (デフォルトを大きく)
        learning_rule: Optional[BioLearningRule] = None
    ):
        super().__init__()

        self.reservoir_size = reservoir_size
        self.input_scale = input_scale

        # 1. Input Synapses
        self.visual_proj = nn.Linear(
            num_visual_inputs, reservoir_size, bias=False)
        self.audio_proj = nn.Linear(
            num_audio_inputs, reservoir_size, bias=False)
        self.text_proj = nn.Linear(num_text_inputs, reservoir_size, bias=False)
        self.somato_proj = nn.Linear(
            num_somato_inputs, reservoir_size, bias=False)

        # 重み初期化 (入力を強力に伝えるためGainを上げる)
        nn.init.uniform_(self.visual_proj.weight, -0.5, 0.5)
        nn.init.uniform_(self.audio_proj.weight, -0.5, 0.5)
        nn.init.uniform_(self.text_proj.weight, -0.5, 0.5)
        nn.init.uniform_(self.somato_proj.weight, -0.5, 0.5)

        # 2. Recurrent Synapses
        self.recurrent_weights = nn.Linear(
            reservoir_size, reservoir_size, bias=False)
        self._init_sparse_weights(self.recurrent_weights, sparsity)

        # 3. Parameters
        self.tau = tau
        self.threshold = threshold
        self.learning_rule = learning_rule

        # 4. State
        self.mem = None
        self.spike = None
        self.prev_spike: Optional[torch.Tensor] = None

    def _init_sparse_weights(self, layer: nn.Linear, sparsity: float) -> None:
        # リザーバのスペクトル半径を調整するイメージで、少し大きめに初期化
        nn.init.normal_(layer.weight, mean=0.0, std=0.5)
        with torch.no_grad():
            mask = (torch.rand_like(layer.weight) < sparsity).float()
            layer.weight.data *= mask

    def forward(
        self,
        visual_spikes: Optional[torch.Tensor] = None,
        audio_spikes: Optional[torch.Tensor] = None,
        text_spikes: Optional[torch.Tensor] = None,
        somato_spikes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        device = self.recurrent_weights.weight.device
        current_input = torch.tensor(0.0, device=device)
        batch_size = 0

        # 入力プロジェクションの計算
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

        # 入力をスケーリングして発火しやすくする
        current_input = current_input * self.input_scale

        if batch_size == 0:
            if self.spike is not None:
                return self.spike
            return torch.zeros(1, self.reservoir_size, device=device)

        # 状態初期化
        mem = self.mem
        spike = self.spike
        if mem is None or spike is None or mem.size(0) != batch_size:
            mem = torch.zeros(batch_size, self.reservoir_size).to(device)
            spike = torch.zeros(batch_size, self.reservoir_size).to(device)

        # リカレント入力
        recurrent_input = self.recurrent_weights(spike)
        self.prev_spike = spike.clone()

        # LIF Dynamics
        total_input = current_input + recurrent_input
        decay = torch.exp(torch.tensor(-1.0 / self.tau, device=device))
        mem = mem * decay + total_input * (1 - decay)

        # Fire
        is_fire = (mem >= self.threshold)
        new_spike = is_fire.float()

        # Reset
        mem = mem - (new_spike * self.threshold)

        # 状態更新
        self.mem = mem
        self.spike = new_spike

        # 学習
        if self.training and self.learning_rule is not None and self.prev_spike is not None:
            self._apply_plasticity(self.prev_spike, new_spike)

        return new_spike

    def _apply_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> None:
        if self.learning_rule is None:
            return

        dw, _ = self.learning_rule.update(
            pre_spikes=pre_spikes,
            post_spikes=post_spikes,
            weights=self.recurrent_weights.weight
        )

        with torch.no_grad():
            self.recurrent_weights.weight += dw
            self.recurrent_weights.weight.clamp_(-1.0, 1.0)

    def reset_state(self) -> None:
        self.mem = None
        self.spike = None
        self.prev_spike = None
