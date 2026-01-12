# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: SNNコア・ラッパー (Robust Ver.)
# 目的: SNNモデルの統一インターフェースを提供し、バックエンドの違いや動的属性アクセスを安全に吸収する。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, cast, Union
import logging
from snn_research.core.base import BaseModel

logger = logging.getLogger(__name__)


class SNNCore(BaseModel):
    """
    SNNモデルの統一インターフェース。
    spikingjelly等のバックエンドを使用するモデルをラップし、
    統計情報の収集やインターフェースの統一を行う。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        vocab_size: int = 1000,
        backend: str = "spikingjelly"
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.backend = backend

        from snn_research.core.architecture_registry import ArchitectureRegistry

        arch_type = self.config.get('architecture_type', 'unknown')
        logger.info(f"Initializing SNNCore with architecture: {arch_type}")

        try:
            # mypyの誤認を防ぐため、敢えてAnyのまま安全に取り扱う
            self.model: Any = ArchitectureRegistry.build(
                arch_type, self.config, vocab_size)
        except Exception as e:
            logger.error(f"Failed to build model '{arch_type}': {e}")
            raise RuntimeError(f"Model build failed: {e}")

        self._init_weights()
        self.spike_stats: Dict[str, float] = {}

    def forward(self, x: Optional[Union[torch.Tensor, Dict[str, Any]]] = None, **kwargs: Any) -> Any:
        """
        順伝播処理。Tensorまたは辞書形式の入力を柔軟に受け付ける。
        """
        # 入力がNoneの場合、kwargsから主要なキーを探す
        if x is None:
            for key in ['input_ids', 'input_images', 'input_sequence', 'x', 'input', 'pixel_values']:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break

        # それでもNoneの場合のガード
        if x is None and not kwargs:
            raise ValueError("SNNCore forward called with no input data.")

        try:
            # self.model(x) の呼び出し自体は nn.Module として許容される
            if x is not None:
                # 入力がDictの場合の展開処理（HuggingFace形式など）
                if isinstance(x, dict):
                    output = self.model(**x, **kwargs)
                else:
                    output = self.model(x, **kwargs)
            else:
                output = self.model(**kwargs)

            self._update_firing_stats()
            return output
        except Exception as e:
            logger.error(f"SNNCore: Forward execution failed: {e}")
            raise e

    def _update_firing_stats(self) -> None:
        """発火率統計を安全に更新する。"""
        # [修正] 直接ドットでアクセスせず、getattrとCallableチェックを使用する
        get_rates_method = getattr(self.model, 'get_firing_rates', None)

        if get_rates_method is not None and callable(get_rates_method):
            try:
                rates = get_rates_method()
                if isinstance(rates, dict):
                    for layer_name, rate in rates.items():
                        val = float(rate.item()) if isinstance(
                            rate, torch.Tensor) else float(rate)
                        self.spike_stats[layer_name] = val
            except Exception:
                pass  # 統計取得の失敗はメイン処理を止めない
        else:
            # 代替案として子モジュールの get_firing_rate を探索
            for name, module in self.model.named_modules():
                child_rate_method = getattr(module, 'get_firing_rate', None)
                if child_rate_method is not None and callable(child_rate_method):
                    try:
                        self.spike_stats[name] = float(child_rate_method())
                    except Exception:
                        pass

    def get_firing_rates(self) -> Dict[str, float]:
        """外部から統計を取得するためのメソッド。"""
        return self.spike_stats

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """[修正] generateメソッドを安全に呼び出す。"""
        gen_method = getattr(self.model, 'generate', None)
        if gen_method is not None and callable(gen_method):
            return cast(torch.Tensor, gen_method(input_ids, **kwargs))

        logger.warning(
            f"Model {type(self.model).__name__} does not support generation. Returning input.")
        return input_ids

    def reset_state(self) -> None:
        """[修正] ネットワーク状態と統計をリセットする。ライブラリ依存を考慮。"""
        # SpikingJellyのリセット機能
        try:
            from spikingjelly.activation_based import functional
            functional.reset_net(self.model)
        except ImportError:
            # spikingjellyがない場合はログのみ出力（あるいは無視）
            pass
        except Exception as e:
            logger.warning(f"SpikingJelly reset failed: {e}")

        # モデル独自のreset_stateを安全に呼び出す
        reset_method = getattr(self.model, 'reset_state', None)
        if reset_method is not None and callable(reset_method):
            try:
                reset_method()
            except Exception as e:
                logger.warning(f"Custom reset_state failed: {e}")

        self.spike_stats.clear()
        logger.debug("SNN state and statistics reset.")

    def _init_weights(self) -> None:
        """重みの初期化。"""
        if isinstance(self.model, nn.Module):
            for m in self.model.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    # パラメータがある場合のみ初期化
                    if hasattr(m, 'weight') and m.weight is not None:
                        try:
                            nn.init.xavier_uniform_(m.weight)
                            if hasattr(m, 'bias') and m.bias is not None:
                                nn.init.zeros_(m.bias)
                        except Exception:
                            # 一部の層（量子化層など）は初期化できない場合がある
                            pass

    def update_plasticity(self, x_input: torch.Tensor, target: torch.Tensor, learning_rate: float = 0.01) -> None:
        """
        [Mypy Fix] 可塑性更新メソッド。内部モデルに委譲する。
        """
        update_method = getattr(self.model, 'update_plasticity', None)
        if update_method is not None and callable(update_method):
            try:
                update_method(x_input, target, learning_rate)
            except Exception as e:
                logger.warning(f"Plasticity update failed: {e}")
        else:
            logger.debug(f"Model {type(self.model).__name__} does not support update_plasticity.")