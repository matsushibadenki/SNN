# ファイルパス: snn_research/hybrid/adapter.py
# Title: ANN-SNN アダプタ層 (膜電位デコーディング標準化)
# Description:
# - ANN（アナログ）ドメインとSNN（スパイク）ドメイン間の情報変換を担う。
# - 修正: SpikesToAnalog のデフォルトを 'mem' (膜電位積分) に変更し、
#   低レイテンシ時の情報損失を防ぐ。

import torch
import torch.nn as nn
from typing import Type, Dict, Any, List, Tuple, Optional
import math

from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron
from snn_research.core.base import BaseModel

class AnalogToSpikes(BaseModel): 
    """
    ANNのアナログ出力をSNNのスパイク入力に変換するアダプタ。
    単純なレートコーディング（アナログ値を電流としてLIFに入力）を実装。
    """
    neuron: nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        time_steps: int,
        neuron_config: Dict[str, Any],
        activation: Type[nn.Module] = nn.Identity # デフォルトは恒等写像
    ) -> None:
        super().__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.time_steps = time_steps
        
        # アナログ特徴量をSNNの入力次元に射影する線形層
        self.projection = nn.Linear(in_features, out_features)
        
        # スパイク生成ニューロン
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        # ニューロンクラスの選択
        neuron_class: Type[nn.Module]
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
        elif neuron_type_str == 'glif':
            neuron_class = GLIFNeuron
            # GLIF固有のパラメータ調整
            if 'gate_input_features' not in neuron_params:
                 neuron_params['gate_input_features'] = out_features
        elif neuron_type_str == 'tc_lif':
            neuron_class = TC_LIF
        elif neuron_type_str == 'dual_threshold':
            neuron_class = DualThresholdNeuron
        else:
            # デフォルト
            neuron_class = AdaptiveLIFNeuron
            
        # パラメータフィルタリング（不要なパラメータを除去して渡す）
        try:
            self.neuron = neuron_class(features=out_features, **neuron_params)
        except TypeError:
            # パラメータが合わない場合のフォールバック（最低限のパラメータで初期化）
            self.neuron = neuron_class(features=out_features)
            
        self.output_act = activation()
    
    def forward(self, x_analog: torch.Tensor, return_full_mems: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x_analog: (B, L, D_in) or (B, D_in)
        Returns:
            spikes: (B, L, T, D_out) or (B, T, D_out)
            mems: (B, L, T, D_out) or (B, T, D_out) or None
        """
        # 1. 次元射影
        x: torch.Tensor = self.projection(x_analog)
        # 活性化関数 (ReLUなど)
        x = self.output_act(x)
        
        # 2. 時間軸の導入
        # (B, ..., D_out) -> (B, ..., T, D_out)
        # unsqueeze(-2) で時間次元を追加し、time_steps回リピート
        # repeatの引数: バッチ次元などは1、時間次元のみT
        repeats = [1] * x.dim()
        repeats.insert(-1, self.time_steps)
        
        x_repeated: torch.Tensor = x.unsqueeze(-2).repeat(*repeats)
        
        # フラット化して時間ループ処理の準備
        # (B, ..., T, D_out) -> (B*...*T, D_out) だと効率が悪いので
        # (B*..., T, D_out) として扱う
        
        # 元の形状を保存
        output_shape = x_repeated.shape
        
        # (Batch_dims, T, D_out) -> (Batch_flattened, T, D_out)
        x_time_batched = x_repeated.reshape(-1, self.time_steps, self.out_features)
        
        # 3. スパイク生成
        # 状態リセット
        if isinstance(self.neuron, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)):
             self.neuron.reset()
             # Statefulモードに設定
             if hasattr(self.neuron, 'set_stateful'):
                 self.neuron.set_stateful(True)

        spikes_history: List[torch.Tensor] = []
        mems_history: List[torch.Tensor] = []
        
        for t in range(self.time_steps):
            current_input = x_time_batched[:, t, :]
            
            # ニューロン実行
            out = self.neuron(current_input)
            
            if isinstance(out, tuple):
                spike_t, mem_t = out
            else:
                spike_t = out
                mem_t = torch.zeros_like(spike_t) # ダミー
                
            spikes_history.append(spike_t)
            if return_full_mems:
                mems_history.append(mem_t)
                
        # Stateful解除
        if hasattr(self.neuron, 'set_stateful'):
             getattr(self.neuron, 'set_stateful')(False)

        # スタックして元の形状に戻す
        spikes_stacked = torch.stack(spikes_history, dim=1) # (Batch_flat, T, D_out)
        spikes_out = spikes_stacked.reshape(output_shape)
        
        mems_out: Optional[torch.Tensor] = None
        if return_full_mems and mems_history:
            mems_stacked = torch.stack(mems_history, dim=1)
            mems_out = mems_stacked.reshape(output_shape)
            
        return spikes_out, mems_out


class SpikesToAnalog(nn.Module):
    """
    SNNのスパイク出力をANNのアナログ入力に変換するアダプタ。
    時間平均（レート）または膜電位積分（Membrane Potential Integration）を行う。
    
    doc/SNN開発：Gemma3:GPT-4のSNN変換に関する技術的考察と戦略.md (5.2) の
    SpikesToAnalog (アグリゲータ) に対応。
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        method: str = "mem", # デフォルトを 'mem' に変更
        tau_mem: float = 20.0 # method="mem" の場合の時定数
    ) -> None:
        """
        Args:
            in_features (int): 入力スパイク特徴量の次元数。
            out_features (int): 出力アナログ特徴量の次元数。
            method (str): 集約方法。"rate" (時間平均) または "mem" (膜電位積分)。
            tau_mem (float): 膜電位積分用の時定数。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method
        self.tau_mem = tau_mem
        
        # スパイク集約結果をアナログ特徴量に射影する線形層
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x_spikes: torch.Tensor) -> torch.Tensor:
        """
        スパイクテンソル (B, ..., T, D_in) をアナログテンソル (B, ..., D_out) に変換する。
        時間次元は最後から2番目 (dim=-2) であることを想定。

        Args:
            x_spikes (torch.Tensor): SNNからのスパイク出力。

        Returns:
            torch.Tensor: ANNに入力するためのアナログ特徴量。
        """
        
        if self.method == "rate":
            # 時間次元 (-2) で平均を取り、発火率を計算
            x_aggregated: torch.Tensor = x_spikes.mean(dim=-2)
            
        elif self.method == "mem":
            # --- 膜電位積分 (Leaky Integrator) の実装 ---
            # スパイク列を入力電流と見なし、非発火LIFニューロンで積分してアナログ値を得る。
            # v[t] = v[t-1] * decay + spike[t]
            
            # 時間次元を取得
            T = x_spikes.shape[-2]
            
            # 減衰係数
            decay = math.exp(-1.0 / self.tau_mem)
            
            # 軸を入れ替えて時間を先頭に持ってくる (T, ..., D_in)
            # dim=-2 を 0 に移動
            dims = list(range(x_spikes.ndim))
            time_dim_idx = x_spikes.ndim - 2
            permute_dims = [time_dim_idx] + dims[:time_dim_idx] + dims[time_dim_idx+1:]
            
            x_time_first = x_spikes.permute(permute_dims)
            
            # 積分用バッファ (時間は除去された形状)
            mem = torch.zeros_like(x_time_first[0])
            
            for t in range(T):
                spike_t = x_time_first[t]
                mem = mem * decay + spike_t
            
            x_aggregated = mem
            
        else:
            raise NotImplementedError(f"Aggregation method '{self.method}' is not implemented.")
            
        # 射影してアナログ特徴量に変換
        x_analog: torch.Tensor = self.projection(x_aggregated)
        
        return x_analog