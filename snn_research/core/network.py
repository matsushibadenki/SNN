# ファイルパス: snn_research/core/network.py
# 修正箇所: メソッド呼び出しの型安全性の確保

# ... (インポート) ...
from snn_research.layers.abstract_layer import AbstractLayer

class AbstractNetwork(nn.Module, ABC):
    # ...
    def build_model(self) -> None:
        for layer in self.layers:
            # layer が AbstractLayer であることを保証し、
            # Tensor オブジェクトと誤認されないようにキャストまたは型確認を行う
            if isinstance(layer, AbstractLayer):
                layer.build() # AbstractLayerにbuild()が定義されていることを確認
        self.built = True

    def update_model(self, inputs: Tensor, targets: Optional[Tensor], model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        all_metrics = {}
        current_input = inputs
        for layer in self.layers:
            if isinstance(layer, AbstractLayer):
                # 明示的に型チェックを通す
                metrics = layer.update_local(current_input, targets, model_state)
                all_metrics.update({f"{layer.name}_{k}": v for k, v in metrics.items()})
                current_input = model_state.get(f'{layer.name}_output', current_input)
        return all_metrics
