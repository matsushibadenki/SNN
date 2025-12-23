# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (HybridNeuromorphicCore)
# 目的: 論理演算ベースのSNNと熱力学的予測レイヤーを統合し、BPなしで自己組織化する知能ユニットを実現する。

import torch
import torch.nn as nn
from typing import Dict, Optional, cast
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN
from snn_research.core.layers.active_predictive_layer import ActivePredictiveLayer

class HybridNeuromorphicCore(nn.Module):
    """
    Hybrid Neuromorphic Core.
    
    構成:
    1. LogicGatedSNN: 高速な感覚処理と特徴抽出 (1.58-bitロジック)。
    2. ActivePredictiveLayer: 抽出された特徴に基づき、内部モデルとの不整合（驚き）を熱力学的に処理。
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        # 1段目: 論理ゲートによるスパイク駆動処理
        self.fast_process = LogicGatedSNN(in_features, hidden_features)
        
        # 2段目: 予測符号化と熱力学サンプリングによる能動的推論
        self.deep_process = ActivePredictiveLayer(hidden_features)
        
        # 出力層 (同様に論理ゲートベース)
        self.output_gate = LogicGatedSNN(hidden_features, out_features)
        
        # 内部状態の記録
        self.last_spikes: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        循環的な推論プロセス:
        感覚入力 -> 論理抽出 -> 熱力学的緩和 -> 出力生成
        """
        # 1. 高速特徴抽出 (Logic Gate)
        # 行列積を排除したビット演算ベースの処理
        features = self.fast_process(x)
        
        # 2. 能動的推論 (Thermodynamic Predictive)
        # 内部モデルとの誤差に基づき、サンプリング温度を調整して「解」を得る
        refined_features = self.deep_process(features)
        
        # 3. 最終出力の生成
        out_spikes = self.output_gate(refined_features)
        
        self.last_spikes = features.detach()
        return out_spikes

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        外部のオプティマイザやBackpropを介さない、完全局所的な学習ステップ。
        """
        with torch.no_grad():
            # 順伝播
            features = self.fast_process(x)
            refined = self.deep_process(features)
            out = self.output_gate(refined)
            
            # 1. 構造的可塑性の更新 (LogicGatedSNNの学習)
            # 入力xと抽出されたfeaturesの相関で学習
            self.fast_process.update_plasticity(x.view(-1), features.view(-1))
            
            # 2. 出力ゲートの更新
            # 目標値がある場合はそれを利用し、ない場合は自己教師あり（予測誤差）を利用
            feedback = target if target is not None else refined
            self.output_gate.update_plasticity(refined.view(-1), feedback.view(-1))
            
            # 3. 予測精度の計算 (メトリクス)
            error_val = 0.0
            if self.deep_process.last_error is not None:
                error_val = float(self.deep_process.last_error.abs().mean().item())
                
        return {
            "prediction_error": error_val,
            "fast_layer_states": float(self.fast_process.states.mean().item())
        }