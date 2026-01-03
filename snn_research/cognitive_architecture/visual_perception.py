# ファイルパス: snn_research/cognitive_architecture/visual_perception.py
# 日本語タイトル: 視覚知覚モジュール (DIコンテナ対応版)
# 修正: __init__引数を追加し、BrainContainerからの注入に対応。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class VisualCortex(nn.Module):
    def __init__(
        self, 
        num_neurons: int = 784, 
        feature_dim: int = 256,
        workspace: Optional[Any] = None,
        vision_model_config: Optional[Any] = None,
        projector_config: Optional[Any] = None,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Args:
            num_neurons: 入力ニューロン数 (Default legacy argument)
            feature_dim: 特徴量次元 (Default legacy argument)
            workspace: GlobalWorkspace (DI injected)
            vision_model_config: 視覚モデル設定 (DI injected)
            projector_config: プロジェクター設定 (DI injected)
            device: デバイス (DI injected)
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.workspace = workspace
        self.device = device
        
        # projector_configが注入された場合はそれを使用、なければデフォルト
        if projector_config:
            # DictまたはObjectとしてのアクセスに対応
            v_dim = 128
            l_dim = 256
            
            if isinstance(projector_config, dict):
                v_dim = projector_config.get('visual_dim', 128)
                l_dim = projector_config.get('lang_dim', 256)
            else:
                # OmegaConfやNamespaceの場合
                v_dim = getattr(projector_config, 'visual_dim', 128)
                l_dim = getattr(projector_config, 'lang_dim', 256)
                
            self.projector: Any = nn.Linear(v_dim, l_dim).to(device)
        else:
            self.projector = nn.Linear(num_neurons, feature_dim).to(device)

    def perceive(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.projector(x)
        return {"features": features}

    def reset_state(self) -> None:
        """状態のリセット。"""
        if hasattr(self.projector, 'reset_state'):
            method = getattr(self.projector, 'reset_state')
            if callable(method):
                method()