# ファイルパス: snn_research/conversion/fold_bn.py
# (修正)
# Title: BatchNorm Folding ユーティリティ
# Description:
# ANNからSNNへの変換精度を向上させるため、Convolution層とそれに続くBatchNorm層を
# 単一のConvolution層に統合（folding）する機能を提供する。
# mypyエラーを完全に解消し、再帰的な探索ロジックを実装。

import torch
import torch.nn as nn
import logging
from typing import cast, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convolution層とBatchNorm層を統合し、新しい重みとバイアスを計算する。

    Args:
        conv (nn.Conv2d): 畳み込み層。
        bn (nn.BatchNorm2d): バッチ正規化層。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 統合後の新しい重みとバイアス。
    """
    w = conv.weight.clone().detach()
    bias = conv.bias.clone().detach() if conv.bias is not None else torch.zeros(w.size(0), device=w.device)

    # mypyエラー[union-attr]を解消: running_mean/varがNoneの場合、foldingは不可能
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError(
            "Cannot fold BatchNorm2d layer because it does not have running stats. "
            "Ensure the model is in eval() mode and track_running_stats was True during training."
        )

    eps = bn.eps
    gamma = bn.weight.clone().detach()
    beta = bn.bias.clone().detach()
    running_mean = bn.running_mean.clone().detach()
    running_var = bn.running_var.clone().detach()

    denom = torch.sqrt(running_var + eps)
    w_fold = w * (gamma / denom).reshape(-1, 1, 1, 1)
    b_fold = (bias - running_mean) * (gamma / denom) + beta
    
    return w_fold, b_fold

def fold_all_batchnorms(model: nn.Module) -> nn.Module:
    """
    モデル内のすべてのConv-BNペアを探索し、インプレースで統合する。
    nn.Sequential内も再帰的に探索する。

    Args:
        model (nn.Module): 変更対象のモデル。

    Returns:
        nn.Module: BatchNormが統合されたモデル。
    """
    model.eval()
    
    # モジュールのトップレベルの子モジュールをイテレート
    for name, module in list(model.named_children()):
        # 1. Sequentialブロック内の探索
        if isinstance(module, nn.Sequential):
            layers = list(module.children())
            for i in range(len(layers) - 1):
                if isinstance(layers[i], nn.Conv2d) and isinstance(layers[i+1], nn.BatchNorm2d):
                    conv = cast(nn.Conv2d, layers[i])
                    bn = cast(nn.BatchNorm2d, layers[i+1])
                    
                    if bn.track_running_stats:
                        logging.info(f"Folding BN in Sequential block '{name}' (layer {i+1}) into Conv (layer {i}).")
                        
                        new_weight, new_bias = fold_conv_bn(conv, bn)
                        
                        new_conv = nn.Conv2d(
                            in_channels=conv.in_channels,
                            out_channels=conv.out_channels,
                            kernel_size=cast(Union[int, tuple[int, int]], conv.kernel_size),
                            stride=cast(Union[int, tuple[int, int]], conv.stride),
                            padding=cast(Union[str, int, tuple[int, int]], conv.padding),
                            dilation=cast(Union[int, tuple[int, int]], conv.dilation),
                            groups=conv.groups,
                            bias=True
                        )
                        new_conv.weight.data.copy_(new_weight)
                        if new_conv.bias is not None:
                            new_conv.bias.data.copy_(new_bias)
                            
                        # Sequentialブロック内のレイヤーを置き換え
                        module[i] = new_conv
                        module[i+1] = nn.Identity()
                    else:
                        logging.warning(f"Skipping folding for BN in '{name}' as track_running_stats is False.")

        # 2. 再帰的に子モジュールに適用
        if len(list(module.children())) > 0:
            fold_all_batchnorms(module)
            
    return model