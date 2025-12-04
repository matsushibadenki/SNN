# ファイルパス: snn_research/core/learning_rule.py
# タイトル: 抽象学習アルゴリズムインターフェース (PyTorch準拠)
# 機能説明: 
#   Project SNN4のロードマップ (P1-4) に基づく、
#   新学習アルゴリズムのコアロジックを抽象化するインターフェース。
#
#   (ダミー実装の解消):
#   - __init__ と _parse_hparams を変更し、
#     具象レイヤー (LIFLayerなど) から 'layer_name' を
#     受け取れるようにします。

from abc import ABC, abstractmethod
from typing import Dict, Any, Iterable, Optional

# --- PyTorch のインポート ---
import torch
from torch import Tensor
import torch.nn as nn

Parameters = Iterable[nn.Parameter]

class AbstractLearningRule(ABC):
    """
    新学習アルゴリズムのための抽象基底クラス (P1-4)。
    """

    def __init__(self, params: Parameters, **kwargs: Any) -> None:
        """
        学習アルゴリズムを初期化します。

        Args:
            params (Parameters): 最適化対象のモデルパラメータ (nn.Parameter)。
            **kwargs: 'learning_rate', 'layer_name' など。
        """
        self.params: Parameters = params
        # 修正: _parse_hparams が self.layer_name も設定する
        self.layer_name: Optional[str] = None
        self.hparams: Dict[str, Any] = self._parse_hparams(kwargs)

    def _parse_hparams(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        hparams: Dict[str, Any] = {}
        hparams['learning_rate'] = kwargs.get('learning_rate', 0.01)
        
        # --- ダミー実装の解消 (P1-1 / P4-4 連携) ---
        # レイヤーが自身の名前を渡せるようにする
        self.layer_name = kwargs.get('layer_name')
        
        return hparams

    @abstractmethod
    def step(
        self,
        inputs: Tensor,
        targets: Optional[Tensor],
        model_state: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        単一の学習ステップ（重み更新）を実行します。
        """
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        """
        勾配バッファをゼロ化します (必要な場合)。
        """
        pass

    def set_learning_rate(self, lr: float) -> None:
        self.hparams['learning_rate'] = lr

    def get_hparams(self) -> Dict[str, Any]:
        return self.hparams

# (型チェックおよびP1-1 具象化のデモンストレーション)
class ExamplePredictiveCodingRule(AbstractLearningRule):
    """
    P1-1 (予測符号化) を実装する具象クラスの例。
    """
    def __init__(self, params: Parameters, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.error_weight: float = float(kwargs.get('error_weight', 0.5))
        self.hparams['error_weight'] = self.error_weight
        # (self.layer_name は super().__init__ で設定される)

    def step(
        self,
        inputs: Tensor,
        targets: Optional[Tensor],
        model_state: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        
        lr: float = float(self.hparams['learning_rate'])
        
        # (ダミー実装の解消)
        # 自身に関連付けられたレイヤー名を使って誤差を取得
        prediction_error: Tensor = torch.tensor(0.0)
        if self.layer_name:
            prediction_error = model_state.get(
                f'prediction_error_{self.layer_name}', 
                torch.tensor(0.0)
            )
        
        update_magnitude: float = 0.0
        
        with torch.no_grad():
            for param in self.params:
                if isinstance(param, nn.Parameter):
                    # (ダミーの更新ロジック)
                    update: Tensor = lr * prediction_error * 0.1
                    param.sub_(update)
                    update_magnitude += float(update.abs().sum().item())
        
        return {'update_magnitude': torch.tensor(update_magnitude)}

    def zero_grad(self) -> None:
        pass

if __name__ == '__main__':
    try:
        import torch
        
        dummy_param_1: nn.Parameter = nn.Parameter(torch.tensor(0.5))
        dummy_params_iterable: Parameters = [dummy_param_1]
        
        rule: AbstractLearningRule = ExamplePredictiveCodingRule(
            dummy_params_iterable, 
            learning_rate=0.001, 
            layer_name="test_layer" # (P1-1)
        )

        dummy_state: Dict[str, Tensor] = {
            'prediction_error_test_layer': torch.tensor(0.5),
        }
        
        metrics: Dict[str, Tensor] = rule.step(
            inputs=torch.tensor(1.0), 
            targets=None, 
            model_state=dummy_state
        )
        
        print(f"Metrics: {metrics}")
        print(f"HParams: {rule.get_hparams()}")
        print(f"Layer Name: {rule.layer_name}")

    except ImportError:
        print("Torch not found, skipping __main__ test.")
