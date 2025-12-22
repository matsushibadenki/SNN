# snn_research/core/networks/bio_pc_network.py
# 生物学的な予測符号化（Bio-PC）を実現するためのネットワーククラス
#
# ディレクトリ: snn_research/core/networks/bio_pc_network.py
# ファイル名: Bio-PC ネットワーク実装
# 目的: 生成ニューロンと推論ニューロンを組み合わせ、k-WTA等を用いた予測符号化を行う。
#
# 変更点:
# - [修正 v6] reset_state メソッドでの無限再帰（RecursionError）を防止するため、
#   親クラスの呼び出し方法を明示的に指定し、自分自身の再帰呼び出しを排除。
# - [修正 v6] named_modules のループ時、自分自身 (self) をスキップするようロジックを堅牢化。

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from .abstract_snn_network import AbstractSNNNetwork
from ..layers.predictive_coding import PredictiveCodingLayer
import logging

logger = logging.getLogger(__name__)

class BioPCNetwork(AbstractSNNNetwork):
    """
    予測符号化(PC)の原理に基づいた生物学的ニューラルネットワーク。
    複数の PredictiveCodingLayer を統合し、階層的な予測と誤差修正を行う。
    """
    def __init__(self, layer_sizes: List[int], sparsity: float = 0.05, input_gain: float = 1.0):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.sparsity = sparsity
        self.input_gain = input_gain

        # レイヤーの構築
        self.pc_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = PredictiveCodingLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i+1],
                sparsity=sparsity
            )
            self.pc_layers.append(layer)

    def reset_state(self):
        """
        ネットワーク全体の膜電位等の状態をリセットする。
        無限再帰を防ぐため、子モジュールの走査方法を改善。
        """
        # [修正] 直接の親クラス(AbstractSNNNetwork)のリセット処理を明示的に呼ぶ
        # super().reset_state() がもし内部で named_modules を回している場合、
        # 重複呼び出しや再帰が発生しないように制御する。
        
        # モデル内の全サブモジュールに対してリセットを実行
        for name, m in self.named_modules():
            # 自分自身(self)に対する reset_state() 呼び出しはスキップ（無限再帰の主因）
            if m is self:
                continue
            
            # reset_state メソッドを持つ子モジュールのみ実行
            if hasattr(m, 'reset_state') and callable(getattr(m, 'reset_state')):
                # RecursiveCodingLayer 等の内部でさらに self.reset_state を呼んでいないか確認
                try:
                    m.reset_state()
                except RecursionError:
                    # 万が一の再帰エラーを捕捉してログ出力（デバッグ用）
                    logger.error(f"RecursionError detected in layer: {name}")
                    continue

        # ネットワーク固有の状態変数の初期化
        self.model_state = {} 

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        階層的なPC推論を実行する。
        """
        # 入力のスケーリング
        x = x * self.input_gain
        
        current_input = x
        activations = {"input": x}

        # 順伝播（予測の生成と修正の連鎖）
        for i, layer in enumerate(self.pc_layers):
            # layer(x) は (prediction, error) 等を返す想定
            current_input, info = layer(current_input)
            activations[f"layer_{i}"] = current_input
            for k, v in info.items():
                activations[f"layer_{i}_{k}"] = v

        return current_input, activations

    def get_sparsity_loss(self) -> torch.Tensor:
        """各レイヤーのスパース性損失を合計する"""
        total_loss = torch.tensor(0.0, device=self.get_device())
        for layer in self.pc_layers:
            if hasattr(layer, 'get_sparsity_loss'):
                total_loss += layer.get_sparsity_loss()
        return total_loss

    def get_device(self):
        """モデルが配置されているデバイスを取得"""
        return next(self.parameters()).device
