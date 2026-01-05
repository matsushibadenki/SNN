# ファイルパス: snn_research/conversion/conversion_utils.py
# (更新)
# Title: ANN-SNN変換 ユーティリティ
# Description:
# ANNからSNNへの変換プロセスにおける性能を最大化するための、
# 高度な正規化およびキャリブレーション技術を提供する。
# 堅牢な重みコピー、パーセンタイルベースの閾値キャリブレーションを実装。
#
# 修正 (v_syn): SyntaxError: unmatched '}' を解消するため、末尾の '}' を削除。

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_copy_weights(target_model: nn.Module, source_state_dict: Dict[str, torch.Tensor], verbose: bool = True) -> Tuple[List[str], List[str]]:
    """
    安全に重みをコピーする。デバイス、データ型をターゲットに合わせ、
    no_gradコンテキストで実行し、不一致キーをログに出力する。

    Returns:
        Tuple[List[str], List[str]]: (missing_keys, unexpected_keys)
    """
    missing_keys, unexpected_keys = [], []
    target_sd = target_model.state_dict()
    copied_count = 0

    with torch.no_grad():
        for k, v_target in target_sd.items():
            if k in source_state_dict:
                v_source = source_state_dict[k]
                if v_source.shape == v_target.shape:
                    v_target.copy_(v_source.to(v_target.device, v_target.dtype))
                    copied_count += 1
                else:
                    if verbose:
                        logging.warning(f"[Shape Mismatch] Key '{k}': target shape {v_target.shape}, source shape {v_source.shape}. Skipped.")
            else:
                missing_keys.append(k)
        
        for k_source in source_state_dict:
            if k_source not in target_sd:
                unexpected_keys.append(k_source)

    if verbose:
        logging.info(f"Weight copy summary: {copied_count} params copied.")
        if missing_keys:
            logging.warning(f"Missing keys in source state_dict: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys in source state_dict: {unexpected_keys}")
            
    return missing_keys, unexpected_keys


@torch.no_grad()
def calibrate_thresholds_by_percentile(
    ann_model: nn.Module, 
    dataloader: Any, 
    percentile: float = 99.9, 
    device: str = "cpu"
) -> Dict[str, float]:
    """
    ANNモデルの各層の活性化を記録し、パーセンタイルに基づいて
    SNNの閾値を決定する。

    Args:
        ann_model (nn.Module): 活性化を記録するANNモデル。
        dataloader: キャリブレーション用データローダー。
        percentile (float): 閾値決定に使用するパーセンタイル。
        device (str): 計算に使用するデバイス。

    Returns:
        Dict[str, float]: レイヤー名をキー、計算された閾値を値とする辞書。
    """
    ann_model.eval()
    ann_model.to(device)
    
    activations: Dict[str, List[torch.Tensor]] = {}

    def get_activation(name: str):
        def hook(model, input, output):
            # ReLUなどの活性化関数の出力を記録
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach())
        return hook

    hooks = []
    for name, module in ann_model.named_modules():
        if isinstance(module, (nn.ReLU, nn.GELU)):
            hooks.append(module.register_forward_hook(get_activation(name)))

    logging.info(f"キャリブレーション用データを {len(dataloader)} バッチ処理します...")
    for batch in tqdm(dataloader, desc="Calibrating Thresholds"):
        inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        ann_model(inputs)

    for hook in hooks:
        hook.remove()

    thresholds: Dict[str, float] = {}
    logging.info("各レイヤーの閾値を計算中...")
    for name, act_list in activations.items():
        all_acts = torch.cat(act_list)
        # 活性化の最大値をチャネルごとに取得
        # (B, C, H, W) -> (B*H*W, C) -> max over non-channel dims
        if all_acts.dim() == 4: # Conv
            all_acts = all_acts.permute(0, 2, 3, 1).reshape(-1, all_acts.shape[1])
        
        # 0より大きい活性化のみを考慮
        all_acts = all_acts[all_acts > 0]
        if all_acts.numel() > 0:
            threshold = torch.quantile(all_acts, q=percentile / 100.0).item()
            thresholds[name] = threshold
            logging.info(f"  - Layer '{name}': Threshold = {threshold:.4f}")

    return thresholds