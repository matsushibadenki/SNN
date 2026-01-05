# ファイルパス: snn_research/modules/reflex_module.py
# Title: Reflex Module (Spinal Cord Circuit) v17.0-Stable
# Description:
#   高速な危険回避（反射）を提供するモジュール。
#   修正: 検出ロジックのロバスト化とデバッグ出力の追加。

import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class ReflexModule(nn.Module):
    """
    脊髄反射・自律神経系を模倣した超高速反応モジュール。
    """
    def __init__(
        self, 
        input_dim: int = 128, 
        action_dim: int = 10,
        latency_constraint_ms: float = 1.0,
        threshold: float = 2.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.threshold = threshold
        
        # 1. Safety Circuit (Hard-coded)
        # バイアスなしで入力信号を直接アクションに結びつける
        self.safety_circuit = nn.Linear(input_dim, action_dim, bias=False)
        
        with torch.no_grad():
            self.safety_circuit.weight.zero_()
            self.safety_circuit.weight[0, 0:10] = 5.0 

        # 2. Learned Reflex
        self.reflex_layer = nn.Linear(input_dim, action_dim)
        self.act_func = nn.ReLU()

        logger.info(f"⚡ Reflex Module initialized (Threshold: {threshold}).")

    def forward(self, x: torch.Tensor) -> Tuple[Optional[int], float]:
        """
        Returns: action_id (int | None), confidence (float)
        """
        # ロバスト化: 入力形状への適応
        # 想定: (Batch, InputDim)
        # 動画などが来た場合: (Batch, Time, Channels, H, W) -> (Batch, InputDim) に強制変換
        
        B = x.shape[0]
        
        # 入力次元が期待と異なる場合、適応処理
        if x.shape[-1] != self.input_dim or x.dim() > 2:
            # 単純化: 全次元で平均を取り、InputDimに合わせて複製またはパディング
            # これは「強い光」や「大きな音」といった総量的な刺激への反射をシミュレート
            
            # 平均強度 (Batch, 1)
            intensity = x.view(B, -1).mean(dim=1, keepdim=True)
            
            # InputDimに拡張 (Batch, InputDim)
            # 全チャンネルに同じ刺激が入ったとみなす
            x_adapted = intensity.expand(B, self.input_dim)
        else:
            x_adapted = x

        # Safety Circuit Check
        safety_signal = self.safety_circuit(x_adapted) # (B, ActionDim)
        max_val, action_idx = torch.max(safety_signal, dim=1)
        
        # item()を呼ぶ前にB=1であることを確認、そうでなければ最大値を取る
        if B > 1:
            val_item = float(max_val.max().item())
            # バッチ内で最も強い反応を示したインデックスを採用
            batch_idx = torch.argmax(max_val)
            act_item = int(action_idx[batch_idx].item())
        else:
            val_item = float(max_val.item())
            act_item = int(action_idx.item())
        
        # 閾値チェック
        if val_item > self.threshold:
            # 視覚入力（動画）などの場合、ログがうるさくなるのでデバッグレベルにする
            # logger.debug(f"⚡ Reflex Triggered! Val: {val_item:.2f}")
            return act_item, val_item

        # Learned Reflex Check
        reflex_signal = self.act_func(self.reflex_layer(x_adapted))
        max_r_val, r_action_idx = torch.max(reflex_signal, dim=1)
        
        if B > 1:
            r_val_item = float(max_r_val.max().item())
            batch_idx_r = torch.argmax(max_r_val)
            r_act_item = int(r_action_idx[batch_idx_r].item())
        else:
            r_val_item = float(max_r_val.item())
            r_act_item = int(r_action_idx.item())
        
        if r_val_item > self.threshold:
             return r_act_item, r_val_item
            
        return None, 0.0
    
    def update_reflex(self, sensory_pattern: torch.Tensor, desired_action: int):
        with torch.no_grad():
            target_vec = torch.zeros(self.action_dim)
            target_vec[desired_action] = 1.0
            
            # 学習時も形状を合わせる
            if sensory_pattern.dim() > 1:
                if sensory_pattern.shape[-1] != self.input_dim:
                    intensity = sensory_pattern.mean()
                    sensory_pattern = intensity.expand(self.input_dim)
                else:
                    sensory_pattern = sensory_pattern.mean(dim=0)
                    
            delta = 0.01 * torch.outer(target_vec, sensory_pattern)
            self.reflex_layer.weight += delta