# ファイルパス: snn_research/models/bio/lif_neuron_legacy.py
# (修正: スパイク統計の記録を追加)
# Title: Leaky Integrate-and-Fire (LIF) ニューロンモデル (Legacy)
# Description: 
#   生物学的学習則のためのシンプルなLIFニューロン。
#   修正: 診断ツールとの互換性のため、total_spikesバッファとresetメソッドを追加。

import torch
import torch.nn as nn

class BioLIFNeuron(nn.Module):
    """生物学的学習則のためのシンプルなLIFニューロン。"""
    def __init__(self, n_neurons: int, neuron_params: dict, dt: float = 1.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.tau_mem = neuron_params['tau_mem']
        self.v_thresh = neuron_params['v_threshold']
        self.v_reset = neuron_params['v_reset']
        self.v_rest = neuron_params['v_rest']
        self.dt = dt
        
        self.register_buffer('voltages', torch.full((n_neurons,), self.v_rest))
        # 修正: 統計用バッファの追加
        self.register_buffer('total_spikes', torch.tensor(0.0))
        self.register_buffer('spikes', torch.zeros(n_neurons)) # 直近のスパイク

    def reset(self):
        """状態のリセット"""
        self.voltages.fill_(self.v_rest)
        self.total_spikes.zero_()
        self.spikes.zero_()

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        if self.voltages.device != input_current.device:
            self.voltages = self.voltages.to(input_current.device)

        # 膜電位の漏れ
        leak = (self.voltages - self.v_rest) / self.tau_mem
        
        # 膜電位の更新
        self.voltages += (-leak + input_current) * self.dt
        
        # 発火
        spikes = (self.voltages >= self.v_thresh).float()
        self.spikes = spikes
        
        # 統計更新
        with torch.no_grad():
            self.total_spikes += spikes.sum()

        # リセット
        self.voltages[spikes.bool()] = self.v_reset
        
        return spikes
