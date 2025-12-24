# ファイルパス: snn_research/core/hybrid_core.py
# 日本語タイトル: 統合ニューロモルフィック・コア (ベクトル教示対応版)
# 内容: ターゲット信号からベクトル報酬を生成し、出力層へ精密なフィードバックを行う。

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from snn_research.core.layers.logic_gated_snn import LogicGatedSNN
# ActivePredictiveLayerは今回は補助的に使用、またはパススルーでも可だが、元の構成を維持
from snn_research.core.layers.active_predictive_layer import ActivePredictiveLayer

class HybridNeuromorphicCore(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        # Hidden層のサイズを引数で受け取るように柔軟化
        self.fast_process = LogicGatedSNN(in_features, hidden_features)
        
        # 深層予測層（今回は補助的な役割として維持）
        self.deep_process = ActivePredictiveLayer(hidden_features)
        
        self.output_gate = LogicGatedSNN(hidden_features, out_features)

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        f = self.fast_process(x_input)
        # Deep processを通すことで時間的文脈などを付与するが、
        # 即時反応タスクでは f をショートカットしても良い。今回は元アーキテクチャに従う。
        r = self.deep_process(f)
        return self.output_gate(r)

    def autonomous_step(self, x_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        自律学習ステップ。
        targetが与えられた場合、出力層に対して「教師ありヘブ学習」を行うためのベクトル信号を生成する。
        """
        # 勾配計算は不要（ヘブ則で更新するため）
        with torch.no_grad():
            f = self.fast_process(x_input)
            r = self.deep_process(f)
            out = self.output_gate(r)
            
            reward_scalar = 0.0
            
            if target is not None:
                # target: (batch_size,) のクラスインデックス
                # 出力層への指導信号ベクトルを作成
                # 正解クラス: +1.0 (強化)
                # 不正解クラス: -0.5 (抑制) -> 間違って発火したニューロンを静める
                
                # バッチサイズ1を想定
                tgt_idx = target.item()
                out_vec = out.view(-1)
                
                # ベースラインは -0.2 (何もしないと少し減衰、発火抑制)
                reward_vector = torch.full_like(out_vec, -0.2)
                
                # 正解ニューロンには強い報酬
                reward_vector[tgt_idx] = 1.5
                
                # 不正解なのに発火してしまったニューロンには強い罰
                # (Output=1 かつ Target!=Output の箇所)
                # すでにベースラインで負になっているが、発火した場合はさらに強く叩くロジックは
                # LogicGatedSNN側で modulation * trace で処理される。
                # ここでは「正解ならプラス、それ以外はマイナス」という意図を込める。
                
                # スカラ報酬（全体の良し悪し）も計算（Hidden層用）
                if out_vec[tgt_idx] > 0.5: # 正解が発火した
                    # 他が発火していないほど高スコア
                    wrong_fires = out_vec.sum() - out_vec[tgt_idx]
                    reward_scalar = 1.0 - (wrong_fires * 0.5)
                else:
                    # 正解が発火していない
                    reward_scalar = -1.0
            else:
                reward_vector = 0.0 # 教師なし
                reward_scalar = 0.0

            # 学習の実行
            
            # Hidden層: 全体的な報酬に基づく強化学習 + 自己組織化
            # 入力パターンに対して、良い結果につながった特徴抽出を強化
            self.fast_process.update_plasticity(x_input.view(-1), f.view(-1), reward=reward_scalar)
            
            # Output層: ベクトル教示信号に基づく教師ありヘブ学習
            # これにより「特定の入力(Hidden発火)」に対して「特定の出力」が結びつく
            self.output_gate.update_plasticity(r.view(-1), out.view(-1), reward=reward_vector)
            
            surprise = float(self.deep_process.last_error.abs().mean().item()) if self.deep_process.last_error is not None else 0.0
            
        return {
            "prediction_error": surprise,
            "reward": reward_scalar,
            "output_spike_count": float(out.sum().item()),
            "proficiency": float(self.output_gate.proficiency.item())
        }
