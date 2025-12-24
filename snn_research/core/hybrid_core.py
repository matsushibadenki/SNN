# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (Fix: Direct Feedback Alignment版)

import torch
import torch.nn as nn
from typing import Dict, Optional
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN
# ActivePredictiveLayerはそのまま利用（インポートエラー回避のため定義があると仮定、あるいはパススルー）
try:
    from snn_research.core.layers.active_predictive_layer import ActivePredictiveLayer
except ImportError:
    # 簡易的な代替クラス（ファイルが見つからない場合用）
    class ActivePredictiveLayer(nn.Module):
        def __init__(self, features): super().__init__(); self.last_error = None
        def forward(self, x): return x

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        # 高速処理層 (隠れ層)
        self.fast_process = LogicGatedSNN(in_features, hidden_features)
        
        # 深層予測層 (ここでは特徴変換として機能)
        self.deep_process = ActivePredictiveLayer(hidden_features)
        
        # 出力ゲート
        self.output_gate = LogicGatedSNN(hidden_features, out_features)
        
        # Feedback Alignment用行列 (固定されたランダム行列)
        # 出力層の誤差を隠れ層に投影するために使用
        self.register_buffer(
            'feedback_matrix', 
            torch.randn(out_features, hidden_features) # (Out, Hidden)
        )

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        f = self.fast_process(x_input)
        r = self.deep_process(f)
        return self.output_gate(r)

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        誤差逆伝播法を使わない自律学習ステップ
        Direct Feedback Alignment (DFA) を使用
        """
        with torch.no_grad():
            # 1. Forward Pass
            f = self.fast_process(x_input)
            r = self.deep_process(f)
            out = self.output_gate(r)
            
            reward_scalar_val = 0.0
            surprise = 0.0
            
            if target is not None:
                # 2. 誤差の計算 (Target - Output)
                # targetはインデックスなのでOne-hot化
                target_onehot = torch.zeros_like(out)
                target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
                
                # エラー信号 (Global Error)
                error = target_onehot - out
                
                # 3. 出力層の学習 (Delta Rule)
                # 出力層には直接のエラー信号を渡す
                self.output_gate.update_plasticity(r, out, reward=error)
                
                # 4. 隠れ層の学習 (Direct Feedback Alignment)
                # 出力の誤差を、固定されたフィードバック行列を通じて隠れ層に伝播させる
                # Error: (Batch, Out)
                # Feedback: (Out, Hidden)
                # Hidden Error: (Batch, Hidden) = Error @ Feedback
                hidden_error = torch.matmul(error, self.feedback_matrix)
                
                # ActivePredictiveLayerを通す前の信号(f)を使って学習
                self.fast_process.update_plasticity(x_input, f, reward=hidden_error)

                # ログ用: 正解率のようなもの
                pred = out.argmax(dim=1)
                acc = (pred == target).float().mean().item()
                reward_scalar_val = acc
                
                if hasattr(self.deep_process, 'last_error') and self.deep_process.last_error is not None:
                    surprise = float(self.deep_process.last_error.abs().mean().item())

        return {
            "prediction_error": surprise,
            "reward": reward_scalar_val, # Accuracy
            "output_spike_count": float(out.sum().item() / x_input.size(0)),
            "proficiency": float(self.output_gate.proficiency.item())
        }
