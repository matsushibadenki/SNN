# ファイルパス: snn_research/core/layers/predictive_coding.py
# 日本語タイトル: 予測符号化レイヤー (Predictive Coding Layer)
# 機能説明: 
#   トップダウンの予測とボトムアップの入力の誤差を計算し、内部状態を更新するSNNレイヤー。
#   Generative Path (状態 -> 予測) と Inference Path (誤差 -> 状態更新) を持つ。
#   重み共有 (Weight Tying) や スパース性制約 (Hard k-WTA) をサポート。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union
import logging

# プロジェクト内のニューロン定義をインポート
# (これらのモジュールが存在することを前提とする)
try:
    from snn_research.core.neurons import (
        AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
        TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron,
        BistableIFNeuron, EvolutionaryLeakLIF
    )
except ImportError:
    # 開発環境等でニューロン定義がない場合のフォールバック（型ヒント用）
    AdaptiveLIFNeuron = Any # type: ignore
    IzhikevichNeuron = Any # type: ignore
    GLIFNeuron = Any # type: ignore
    TC_LIF = Any # type: ignore
    DualThresholdNeuron = Any # type: ignore
    ScaleAndFireNeuron = Any # type: ignore
    BistableIFNeuron = Any # type: ignore
    EvolutionaryLeakLIF = Any # type: ignore

logger = logging.getLogger(__name__)

class PredictiveCodingLayer(nn.Module):
    """
    Predictive Coding (PC) を実行するSNNレイヤー。
    入力(Bottom-Up)と予測(Top-Down)の誤差を計算し、次層への出力と自己の状態更新を行う。
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int, 
        neuron_class: Type[nn.Module], 
        neuron_params: Dict[str, Any],
        weight_tying: bool = True,
        sparsity: float = 0.05
    ):
        super().__init__()
        self.weight_tying = weight_tying
        self.sparsity = sparsity
        
        # ニューロンパラメータのフィルタリング（不要な引数を除外）
        filtered_params = self._filter_params(neuron_class, neuron_params)

        # 1. Generative Path (Top-Down: State -> Prediction)
        # 内部状態(d_state)から、下の層の入力次元(d_model)を予測する
        self.generative_fc = nn.Linear(d_state, d_model)
        self.generative_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], 
                                      neuron_class(features=d_model, **filtered_params))
        
        # 2. Inference Path (Bottom-Up: Error -> State Update)
        # 予測誤差から内部状態を更新する
        if self.weight_tying:
            self.inference_fc = None # generative_fcの転置行列を使用
        else:
            self.inference_fc = nn.Linear(d_model, d_state)
            
        self.inference_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], 
                                     neuron_class(features=d_state, **filtered_params))
        
        # 正規化層
        self.norm_state = nn.LayerNorm(d_state)
        self.norm_error = nn.LayerNorm(d_model)
        
        # 学習可能なスケーリング係数
        self.error_scale = nn.Parameter(torch.tensor(1.0))
        self.feedback_strength = nn.Parameter(torch.tensor(1.0))

    def _filter_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        """指定されたニューロンクラスが受け入れるパラメータのみを抽出する"""
        valid_params: List[str] = []
        
        # 型チェックとパラメータリストの定義
        # Note: 実際の実装ではニューロンクラス自体に valid_params プロパティを持たせる設計が望ましい
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset']
        elif neuron_class == IzhikevichNeuron:
            valid_params = ['features', 'a', 'b', 'c', 'd', 'dt']
        elif neuron_class == GLIFNeuron:
            valid_params = ['features', 'base_threshold', 'gate_input_features']
        elif neuron_class == TC_LIF:
            valid_params = ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
        elif neuron_class == DualThresholdNeuron:
            valid_params = ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
        elif neuron_class == ScaleAndFireNeuron:
            valid_params = ['features', 'num_levels', 'base_threshold']
        elif neuron_class == BistableIFNeuron:
            valid_params = ['features', 'v_threshold_high', 'v_reset', 'tau_mem', 'bistable_strength', 'v_rest', 'unstable_equilibrium_offset']
        elif neuron_class == EvolutionaryLeakLIF:
            valid_params = ['features', 'initial_tau', 'v_threshold', 'v_reset', 'detach_reset', 'learn_threshold']
        else:
             # デフォルトパラメータ
             valid_params = ['features', 'tau_mem', 'base_threshold', 'v_reset']
        
        return {k: v for k, v in neuron_params.items() if k in valid_params}

    def _apply_lateral_inhibition(self, x: torch.Tensor) -> torch.Tensor:
        """Hard k-WTA: 絶対値の上位k%のみを残し、他を抑制する（スパース化）"""
        if self.sparsity >= 1.0 or self.sparsity <= 0.0:
            return x
            
        x_abs = x.abs()
        B, N = x.shape
        k = int(N * self.sparsity)
        if k == 0: k = 1
        
        topk_values, _ = torch.topk(x_abs, k, dim=1)
        threshold = topk_values[:, -1].unsqueeze(1)
        # ゼロ除算や極小値回避
        threshold = torch.max(threshold, torch.tensor(1e-6, device=x.device))
        mask = (x_abs >= threshold).float()
        
        return x * mask

    def forward(
        self, 
        bottom_up_input: torch.Tensor, 
        top_down_state: torch.Tensor,
        top_down_error: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            bottom_up_input: 下位層からの入力（またはターゲット）
            top_down_state: 現在の層の内部状態（予測の源）
            top_down_error: 上位層からの予測誤差フィードバック（オプション）
        
        Returns:
            updated_state: 更新された内部状態
            error: この層の予測誤差 (次層への入力となる)
            combined_mem: 可視化用の膜電位
        """
        # 1. Generative Pass: 状態から入力を予測
        # State -> Prediction
        pred_input = self.generative_fc(self.norm_state(top_down_state))
        pred, gen_mem = self.generative_neuron(pred_input)
        
        # 2. Error Calculation: 予測と実際の入力の差
        # Error = Input - Prediction
        raw_error = bottom_up_input - pred
        error = raw_error * self.error_scale
        
        # 3. Inference Pass: 誤差を使って状態を修正
        # Error -> State Update
        norm_error = self.norm_error(error)
        
        if self.weight_tying:
            # 重み共有: 生成重みの転置を使用
            # self.generative_fc.weight の形状は (d_model, d_state)
            # F.linear(input, weight) は input @ weight.T を計算
            # ここでは (B, d_model) @ (d_model, d_state) = (B, d_state) としたい
            # F.linear(norm_error, self.generative_fc.weight.t()) 
            # => norm_error @ (weight.T).T => norm_error @ weight => Shape OK
            bu_input = F.linear(norm_error, self.generative_fc.weight.t())
        else:
            if self.inference_fc is None:
                 raise RuntimeError("inference_fc is None but weight_tying is False")
            bu_input = self.inference_fc(norm_error)

        # フィードバックがある場合は統合
        total_input = bu_input
        if top_down_error is not None:
            total_input = total_input - (top_down_error * self.feedback_strength)
        
        # 推論ニューロンによる状態更新の計算
        state_update, inf_mem = self.inference_neuron(total_input)
        
        # スパース性制約 (k-WTA)
        state_update = self._apply_lateral_inhibition(state_update)
        
        # 状態の更新 (リーク積分のような更新)
        # state(t+1) = 0.9 * state(t) + 0.1 * update
        updated_state = top_down_state * 0.9 + state_update * 0.1
        
        combined_mem = torch.cat((gen_mem, inf_mem), dim=1) 
        
        return updated_state, error, combined_mem