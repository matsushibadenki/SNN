# ファイルパス: snn_research/conversion/optimization_strategies.py
# Title: SNN変換最適化戦略 (Deep Bio-Calibration Logic) - 型修正版
# Description:
#   Roadmap v14.0 "Deep Bio-Calibration" を支える最適化ロジック群。
#   mypyのエラー（Tensor呼び出し、辞書型推論）を修正。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, cast

logger = logging.getLogger(__name__)

class AdaptiveTimestepScheduler:
    """
    層の深さや種類に応じて、シミュレーションタイムステップ数(T)を動的に割り当てるスケジューラ。
    入力層に近いほど高い時間分解能（大きなT）を割り当て、
    深層では抽象度が高まるためTを減らすなどの戦略をとる。
    """
    def __init__(self, base_timesteps: int = 4, max_timesteps: int = 16, strategy: str = "linear_decay"):
        self.base_timesteps = base_timesteps
        self.max_timesteps = max_timesteps
        self.strategy = strategy

    def get_timesteps_for_layer(self, layer_index: int, total_layers: int, layer_type: str) -> int:
        """
        指定された層に推奨されるタイムステップ数を計算する。
        """
        if total_layers <= 0:
            return self.base_timesteps

        # 相対的な深さ (0.0 - 1.0)
        depth_ratio = layer_index / total_layers

        if self.strategy == "linear_decay":
            # 深くなるにつれてタイムステップを減らす（入力の過渡応答を重視）
            # T = Max - (Max - Base) * depth
            t = self.max_timesteps - (self.max_timesteps - self.base_timesteps) * depth_ratio
        
        elif self.strategy == "pyramid":
            # 中間層で最もタイムステップを多くする
            if depth_ratio < 0.5:
                t = self.base_timesteps + (self.max_timesteps - self.base_timesteps) * (depth_ratio * 2)
            else:
                t = self.max_timesteps - (self.max_timesteps - self.base_timesteps) * ((depth_ratio - 0.5) * 2)
        
        else: # constant
            t = self.base_timesteps

        # Conv層は空間情報を時間情報に変換するため、少し多めに確保
        if "Conv" in layer_type:
            t = min(self.max_timesteps, t * 1.2)

        return max(1, int(t))

class ProgressiveQuantization:
    """
    段階的な量子化スケジュールを管理する。
    変換初期は高精度で、徐々にターゲットビット数（例：1.58bit）へ落としていく。
    """
    def __init__(self, stages: int = 5, initial_bits: float = 8.0, target_bits: float = 1.58):
        self.stages = stages
        self.initial_bits = initial_bits
        self.target_bits = target_bits
        self.current_stage = 0

    def get_quantization_level(self) -> float:
        """現在のステージにおける量子化ビット数を返す"""
        if self.current_stage >= self.stages:
            return self.target_bits
        
        # 指数関数的減衰でビット数を減らす
        decay = (self.target_bits / self.initial_bits) ** (1 / max(1, self.stages - 1))
        bits = self.initial_bits * (decay ** self.current_stage)
        
        # ビット数は一般的に整数または特定の値(1.58)をとるため丸める
        if bits > 4:
            return 8.0
        elif bits > 2:
            return 4.0
        elif bits > 1.8:
            return 2.0
        else:
            return 1.58

    def step(self):
        self.current_stage += 1

class LayerWiseOptimizer:
    """
    各層の統計情報（重みの分布、活性化のスパース性）を分析し、
    最適なSNNニューロンタイプとパラメータ設定を提案する。
    """
    def __init__(self) -> None:
        self.strategies: Dict[str, str] = {}

    def analyze_layer(self, layer: nn.Module, layer_name: str, activations: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        層を分析し、推奨設定を返す。
        """
        # --- 修正: 辞書の型を明示的に指定 ---
        recommendation: Dict[str, Any] = {
            "neuron_type": "lif", # default
            "params_hint": {}
        }
        
        # 重みの分析
        if hasattr(layer, 'weight') and layer.weight is not None:
            # --- 修正: layer.weight を torch.Tensor にキャスト ---
            w: torch.Tensor = cast(torch.Tensor, layer.weight)
            
            # --- 修正: Tensorのメソッド呼び出し ---
            w_mean = w.mean().item()
            w_std = w.std().item()
            
            # 重みの分散が大きい場合 -> ダイナミックレンジの広いニューロン (Izhikevich)
            if w_std > 1.5:
                recommendation["neuron_type"] = "izhikevich"
                recommendation["params_hint"] = {"a": 0.02, "b": 0.2} # バースト発火モード
            
            # 重みが極端に小さい場合 -> 閾値を下げるか、感度を上げる
            if abs(w_mean) < 0.01 and w_std < 0.1:
                # --- 修正: 型ヒントにより params_hint への代入が可能に ---
                recommendation["params_hint"]["base_threshold"] = 0.5 # 低閾値
        
        # 活性化の分析 (キャリブレーションデータがある場合)
        if activations is not None:
            # スパース性 (ゼロの割合)
            sparsity = (activations == 0).float().mean().item()
            max_val = activations.max().item()
            
            # 非常にスパースな場合 -> Dual Threshold や TC-LIF でノイズ除去
            if sparsity > 0.8:
                recommendation["neuron_type"] = "dual_threshold"
                recommendation["params_hint"]["threshold_low"] = max_val * 0.1
                recommendation["params_hint"]["threshold_high"] = max_val * 0.8
            
            # 活性化が常に高い場合 -> GLIF (Gated LIF) で抑制
            elif sparsity < 0.1:
                recommendation["neuron_type"] = "glif"
        
        # --- 修正: 値を str にキャスト ---
        self.strategies[layer_name] = cast(str, recommendation["neuron_type"])
        return recommendation

    def get_strategy(self, layer_name: str) -> str:
        return self.strategies.get(layer_name, "lif")