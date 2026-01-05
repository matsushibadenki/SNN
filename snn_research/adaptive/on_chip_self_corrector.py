# ファイルパス: snn_research/adaptive/on_chip_self_corrector.py
# (Phase 4: Autonomous Adaptation)
# Title: On-Chip Self-Correction Module
# Description:
#   推論時にリアルタイムでモデルの状態を監視し、自己修正を行う。
#   - エントロピーに基づく不確実性検知
#   - 恒常性維持（Homeostasis）による閾値調整
#   - 局所的な重み更新（STDP/Hebbian）の動的トリガー

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, cast

class OnChipSelfCorrector(nn.Module):
    """
    オンチップ自己修正・適応モジュール。
    推論モード(eval)であっても、内部状態を監視して適応的なパラメータ更新を行う。
    """
    def __init__(
        self,
        monitor_layers: List[nn.Module],
        adaptation_rate: float = 0.001,
        entropy_threshold: float = 0.6,
        homeostasis_target: float = 0.05  # 目標スパイク率
    ):
        super().__init__()
        self.monitor_layers = nn.ModuleList(monitor_layers)
        self.adaptation_rate = adaptation_rate
        self.entropy_threshold = entropy_threshold
        self.homeostasis_target = homeostasis_target
        
        # 統計情報のバッファ
        self.register_buffer("global_surprise", torch.tensor(0.0))
        self.register_buffer("adaptation_count", torch.tensor(0))

    def _calculate_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """予測分布のエントロピーを計算"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy

    def update_homeostasis(self):
        """
        恒常性維持: 各ニューロンの発火率を目標値に近づけるよう閾値を調整
        （バックプロパゲーション不要の局所更新）
        """
        with torch.no_grad():
            for layer in self.monitor_layers:
                # AdaptiveLIFNeuron などを想定
                # [Fix] Cast to Any to access dynamic attributes
                layer_any = cast(Any, layer)
                if hasattr(layer_any, "avg_firing_rate") and hasattr(layer_any, "base_threshold"):
                    current_rate = layer_any.avg_firing_rate
                    error = current_rate - self.homeostasis_target
                    
                    # 発火しすぎ -> 閾値を上げる (+ error)
                    # 発火しなさすぎ -> 閾値を下げる (- error)
                    # 不感帯を設ける
                    mask = (error.abs() > (self.homeostasis_target * 0.2)).float()
                    delta = error * self.adaptation_rate * mask
                    
                    layer_any.base_threshold.data += delta
                    layer_any.base_threshold.data.clamp_(min=0.1)

    def trigger_plasticity(self, layer: nn.Module, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        驚き（Surprise）が高い場合に可塑性（重み更新）をトリガーする
        簡易的なSTDP/Hebbianルールの適用
        """
        with torch.no_grad():
            # [Fix] Cast to Any to access .weight
            layer_any = cast(Any, layer)
            if hasattr(layer_any, "weight") and layer_any.weight.requires_grad:
                # Hebbian: Fire together, wire together
                # ΔW = η * (Post * Pre^T)
                # バッチ処理のための簡易計算
                if pre_spikes.dim() == 2 and post_spikes.dim() == 2:
                    delta_w = torch.matmul(post_spikes.t(), pre_spikes)
                    
                    # 重みの大きさに応じた正規化 (Oja's rule like)
                    # 実際には勾配を使わず、in-placeで値を更新
                    # [Fix] Explicit cast to Tensor for accumulation
                    layer_any.weight.data += self.adaptation_rate * 0.1 * delta_w
                    
                    # 重みの発散を防ぐための減衰
                    layer_any.weight.data *= 0.999

    def forward(self, logits: torch.Tensor, hidden_states: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """
        推論ステップごとに呼び出される。
        Args:
            logits: モデルの最終出力 (B, Classes)
            hidden_states: 各層の (入力スパイク, 出力スパイク) のリスト
        """
        stats = {}
        
        # 1. 不確実性（エントロピー）の監視
        entropy = self._calculate_entropy(logits)
        stats["entropy"] = entropy.item()
        
        # 2. 「驚き」の検知 (High Entropy = Surprise)
        is_surprised = entropy > self.entropy_threshold
        
        if is_surprised:
            self.global_surprise = entropy
            # [Fix] Explicit cast to Tensor for accumulation
            cast(torch.Tensor, self.adaptation_count).add_(1)
            
            # 3. 適応ステップの実行
            self.update_homeostasis()
            
            # 層ごとの可塑性トリガー (監視対象層とhidden_statesが対応している前提)
            # ここでは簡易的に実装
            pass 
            
        return stats