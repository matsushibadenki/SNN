# ファイルパス: snn_research/cognitive_architecture/thalamus.py
# Title: Thalamus (視床) Module
# Description:
# - 感覚情報の皮質へのリレーとゲーティングを行う。
# - 皮質からのトップダウン注意信号に基づき、入力の選別を行う。
# - Phase 6: 全脳アーキテクチャの重要コンポーネント。

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class Thalamus(nn.Module):
    """
    視床モデル。
    感覚器からのボトムアップ信号と、大脳皮質/前頭前野からのトップダウン信号を統合する。
    """
    def __init__(self, input_dim: int = 784, output_dim: int = 256, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 感覚中継核 (Relay Nuclei): 単純な線形変換ではなく、ゲート付きリレー
        self.relay_weights = nn.Linear(input_dim, output_dim, bias=False)
        
        # 網様体核 (TRN: Thalamic Reticular Nucleus): 抑制性制御
        # トップダウン注意信号を受け取り、リレー細胞を抑制/脱抑制する
        self.attention_gate = nn.Linear(output_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # 状態保持
        self.current_state = "OPEN" # OPEN, GATED, SLEEP
        self.to(device)
        logger.info(f"🧠 Thalamus initialized (In: {input_dim}, Out: {output_dim})")

    def forward(self, sensory_input: torch.Tensor, top_down_attention: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            sensory_input: 感覚器からの生スパイクまたはレート信号
            top_down_attention: 皮質からの注意制御信号 (Optional)
        Returns:
            Dict containing 'relayed_output' and 'gate_status'
        """
        # 1. 基本的なリレー処理
        relayed = self.relay_weights(sensory_input)
        
        # 2. ゲーティング (Attention Control)
        gate_value = torch.ones_like(relayed)
        
        if top_down_attention is not None:
            # 注意信号がある場合、TRNを介してゲートを調整
            # 注意信号が高いほど、ゲートが開く (脱抑制)
            gate_control = self.attention_gate(top_down_attention)
            gate_value = self.sigmoid(gate_control)
            relayed = relayed * gate_value
        
        # 3. 睡眠時の遮断 (Burst mode vs Tonic mode simulation)
        if self.current_state == "SLEEP":
            # 睡眠紡錘波 (Sleep Spindles) のようなノイズのみを通すか、完全に遮断
            relayed = relayed * 0.1 # 大幅に減衰
            
        return {
            "relayed_output": relayed,
            "gate_value": gate_value
        }

    def set_state(self, state: str):
        """
        脳の状態に合わせて視床のモードを変更 (AWAKE / SLEEP)
        """
        if state in ["AWAKE", "SLEEP"]:
            self.current_state = state
            logger.info(f"Thalamus state switched to: {state}")
        else:
            logger.warning(f"Invalid thalamus state: {state}")