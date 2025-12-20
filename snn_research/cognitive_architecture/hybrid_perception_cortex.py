# ファイルパス: snn_research/cognitive_architecture/hybrid_perception_cortex.py
# (Phase 3: Cortical Column Integrated - mypy修正版)
# Title: ハイブリッド知覚野 (Cortical Column + SOM)
# Description:
# - 入力スパイクを「皮質カラム (Cortical Column)」で処理し、階層的な特徴変換を行う。
# - カラムの出力を「自己組織化マップ (SOM)」に入力し、トポロジカルな分類を行う。
# - 修正: mypyエラー (None not callable, assignment type mismatch) を解消。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, cast

from .som_feature_map import SomFeatureMap
from .global_workspace import GlobalWorkspace
from snn_research.core.cortical_column import CorticalColumn

class HybridPerceptionCortex(nn.Module):
    """
    皮質カラムによる階層的処理と、SOMによるクラスタリングを統合した知覚野。
    """
    # クラスレベルまたは__init__内で型を明示
    column: CorticalColumn
    prev_column_state: Optional[Dict[str, torch.Tensor]]

    def __init__(
        self, 
        workspace: GlobalWorkspace, 
        num_neurons: int, 
        feature_dim: int = 64, 
        som_map_size=(8, 8), 
        stdp_params: Optional[Dict[str, Any]] = None,
        cortical_column: Optional[CorticalColumn] = None 
    ):
        super().__init__()
        self.workspace = workspace
        self.num_neurons = num_neurons
        self.feature_dim = feature_dim
        
        # 1. 皮質カラム (CorticalColumn)
        # 注入されない場合は内部で簡易作成するが、self.column は常に CorticalColumn 型とする
        if cortical_column is None:
            print("⚠️ Warning: CorticalColumn not injected. Using fallback.")
            self.column = CorticalColumn(
                input_dim=num_neurons,
                column_dim=feature_dim,
                output_dim=feature_dim,
                neuron_config={'type': 'lif', 'tau_mem': 20.0, 'base_threshold': 1.0}
            )
        else:
            self.column = cortical_column
        
        # 2. 特徴射影 (カラム出力をSOM次元へ)
        self.input_projection = nn.Linear(feature_dim, feature_dim)
        
        if stdp_params is None:
            stdp_params = {'learning_rate': 0.005, 'a_plus': 1.0, 'a_minus': 1.0, 'tau_trace': 20.0}
        
        # 3. 自己組織化マップ (SOM)
        self.som = SomFeatureMap(
            input_dim=feature_dim,
            map_size=som_map_size,
            stdp_params=stdp_params
        )
        
        # カラムの状態保持用 (型を明示)
        self.prev_column_state = None
        
        print("🧠 ハイブリッド知覚野 (Cortical Column + SOM) が初期化されました。")

    def perceive_and_upload(self, spike_pattern: torch.Tensor) -> None:
        """
        入力スパイク -> カラム処理 -> SOM学習 -> Workspaceアップロード
        """
        # spike_pattern: (Time, Neurons) または (Neurons,)
        if spike_pattern.dim() == 2:
            # 時間平均をとってレート入力とする (簡易化)
            input_signal = spike_pattern.float().mean(dim=0).unsqueeze(0) # (1, Neurons)
        else:
            input_signal = spike_pattern.float().unsqueeze(0)

        # 1. 皮質カラムによる処理
        # self.column は CorticalColumn 型であることが保証されているため、呼び出し可能
        out_ff, out_fb, current_states = self.column(input_signal, self.prev_column_state)
        
        # 状態更新 (次のステップのために保持)
        # Dict[str, Tensor] として保存
        self.prev_column_state = {k: v.detach() for k, v in current_states.items()}

        # 2. 特徴射影
        # カラム出力 (1, FeatureDim) -> (FeatureDim,)
        column_output = out_ff.squeeze(0)
        feature_vector = torch.relu(self.input_projection(column_output))

        # 3. SOMによる特徴分類と学習
        # 特徴ベクトルをSOMに入力
        for _ in range(3): # 学習反復
            som_spikes = self.som(feature_vector)
            self.som.update_weights(feature_vector, som_spikes)
        
        final_som_activation = self.som(feature_vector)
        
        # 顕著性スコア (カラムの活性度 + 入力強度)
        column_activity = sum(t.mean().item() for t in current_states.values()) / len(current_states)
        input_strength = input_signal.mean().item()
        salience = min(1.0, (column_activity + input_strength) * 5.0)
        
        perception_data = {
            "type": "perception", 
            "features": final_som_activation,
            "details": f"Processed via Cortical Column (Activity: {column_activity:.2f})"
        }

        self.workspace.upload_to_workspace(
            source="perception",
            data=perception_data,
            salience=salience
        )
        print(f"  - 知覚野: 皮質カラム処理完了 (活性度: {column_activity:.2f}) -> Workspaceへ送信")