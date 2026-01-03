# ファイルパス: snn_research/conversion/ecl_components.py
# (新規作成)
# Title: エラー補償学習 (ECL) コンポーネント
# Description:
# SNN5改善レポート (セクション3.1, 引用[6]) に基づき、
# ANN-SNN変換の精度を向上させるためのECLコンポーネントを定義する。
# - LearnableClippingFunction: 学習可能なしきい値を持つクリッピング関数
#
# 修正 (v2): mypy [name-defined], [assignment] エラーを修正。

import torch
import torch.nn as nn
# --- ▼ 修正: Dict, Any, List, Tuple, Optional をインポート ▼ ---
from typing import Dict, Any, List, Tuple, Optional
# --- ▲ 修正 ▲ ---

class LearnableClippingFunction(torch.autograd.Function):
    """
    ECL (引用[6]) のための、学習可能なしきい値を持つクリッピング関数 (AutoGrad Function)。
    順伝播: y = clamp(x, 0, threshold)
    逆伝播: 勾配をスルー (Straight-Through Estimator, STE)
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctx (Any): コンテキスト
            x (torch.Tensor): 入力 (ANNの活性化)
            threshold (torch.Tensor): 学習可能なしきい値 (スカラーまたはxの形状にブロードキャスト可能)
        """
        # 順伝播では、勾配計算のために threshold を保存
        ctx.save_for_backward(x, threshold)
        # clamp(min=0) はReLUに相当。ここでは threshold による上限クリッピングが主
        return torch.clamp(x, min=0.0, max=threshold.item())

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        逆伝播。
        """
        x, threshold = ctx.saved_tensors
        
        # --- 入力 (x) への勾配 (STE) ---
        # 0 <= x <= threshold の範囲でのみ勾配を流す (STE)
        pass_through_mask = (x >= 0) & (x <= threshold)
        grad_x = grad_output * pass_through_mask.float()
        
        # --- しきい値 (threshold) への勾配 ---
        # 引用[6]の式(5)に基づく:
        # dL/d(threshold) = dL/dy * dy/d(threshold)
        # y = x (if x < threshold), y = threshold (if x >= threshold)
        # dy/d(threshold) = 0 (if x < threshold), 1 (if x >= threshold)
        
        # x が threshold を超えた場合のみ、勾配が流れる
        above_threshold_mask = (x >= threshold)
        grad_threshold = grad_output * above_threshold_mask.float()
        
        # 全バッチ・全ニューロンで勾配を合計する
        grad_threshold_sum = torch.sum(grad_threshold)
        
        return grad_x, grad_threshold_sum

class LearnableClippingLayer(nn.Module):
    """
    ECL (引用[6]) のための学習可能なしきい値を持つクリッピングレイヤー。
    ANNのReLUの代わりに使用する。
    """
    def __init__(self, initial_threshold: float = 1.0, num_features: Optional[int] = None):
        """
        Args:
            initial_threshold (float): しきい値の初期値。
            num_features (Optional[int]): 特徴量数 (チャネルごとなど)。
                                        Noneの場合はスカラーしきい値を使用。
        """
        super().__init__()
        if num_features is not None:
            # チャネルごとの学習可能なしきい値
            self.threshold = nn.Parameter(torch.full((num_features,), initial_threshold))
        else:
            # スカラーしきい値
            self.threshold = nn.Parameter(torch.tensor(initial_threshold))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 形状を合わせる (例: (B, C, H, W) と (C,))
        
        # --- ▼ 修正: [assignment] エラーを修正 (nn.Parameter を上書きしない) ▼ ---
        threshold_expanded: torch.Tensor = self.threshold # デフォルト
        
        if x.dim() == 4 and self.threshold.dim() == 1 and x.shape[1] == self.threshold.shape[0]:
            # (C,) -> (1, C, 1, 1)
            threshold_expanded = self.threshold.view(1, -1, 1, 1)
        elif x.dim() == 2 and self.threshold.dim() == 1 and x.shape[1] == self.threshold.shape[0]:
            # (B, C) と (C,)
            threshold_expanded = self.threshold.view(1, -1)
        
        return LearnableClippingFunction.apply(x, threshold_expanded)
        # --- ▲ 修正 ▲ ---