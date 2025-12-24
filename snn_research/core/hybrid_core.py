# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: Adam最適化内蔵版)

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN
from snn_research.core.layers.active_predictive_layer import ActivePredictiveLayer

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fast_process = LogicGatedSNN(in_features, hidden_features)
        self.deep_process = ActivePredictiveLayer(hidden_features)
        self.output_gate = LogicGatedSNN(hidden_features, out_features)
        
        # 最適化アルゴリズムを内蔵 (Learning Rateはタスクに合わせて調整)
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        f = self.fast_process(x_input)
        r = self.deep_process(f)
        return self.output_gate(r)

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        自律学習ステップ: Forward -> Loss -> Backward -> Optimizer Step
        """
        loss_val = 0.0
        acc_val = 0.0
        
        if target is not None:
            # 1. 勾配リセット
            self.optimizer.zero_grad()
            
            # 2. Forward pass (勾配記録)
            f = self.fast_process(x_input)
            r = self.deep_process(f)
            out = self.output_gate(r)
            
            # 3. Loss計算
            # outはスパイク(0/1)の束なので、logitsとして扱うために少しスケーリングするか、
            # そのまま確率として扱う。ここではCrossEntropyのためにlogitsとみなす。
            # しかしSNN出力は[0,1]なので、少しゲインを掛けてsoftmaxにかかりやすくする。
            logits = out * 10.0 
            loss = self.loss_fn(logits, target)
            
            # 4. Backward & Optimize
            loss.backward()
            
            # 勾配クリッピング (安定化のため)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            
            loss_val = loss.item()
            
            with torch.no_grad():
                pred = out.argmax(dim=1)
                acc_val = (pred == target).float().mean().item()
                out_spike_count = out.sum().item() / x_input.size(0)
        else:
            # 推論のみ
            with torch.no_grad():
                out = self.forward(x_input)
                out_spike_count = out.sum().item() / x_input.size(0)

        return {
            "prediction_error": loss_val,
            "reward": acc_val, # Accuracyを報酬として返す
            "output_spike_count": out_spike_count,
            "proficiency": float(self.output_gate.proficiency.item())
        }
