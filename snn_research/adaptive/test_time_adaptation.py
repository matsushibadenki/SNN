# ファイルパス: snn_research/adaptive/test_time_adaptation.py
# Title: 推論時適応 (Test-Time Adaptation) モジュール
# Description:
#   推論中に教師なし学習則を適用して重みを動的に更新するラッパー。

import torch
import torch.nn as nn
from typing import Any, Optional, List, Type
import logging
from torch.utils.hooks import RemovableHandle

from snn_research.learning_rules.base_rule import BioLearningRule

logger = logging.getLogger(__name__)


class TestTimeAdaptationWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        learning_rule: BioLearningRule,
        target_layers: Optional[List[str]] = None,
        adaptation_rate_multiplier: float = 0.1
    ):
        super().__init__()
        self.model = model
        self.learning_rule = learning_rule
        self.target_layers = target_layers
        self.adaptation_rate_multiplier = adaptation_rate_multiplier
        self.hooks: List[RemovableHandle] = []

        logger.info("🧬 Test-Time Adaptation (TTA) Wrapper initialized.")

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # 1. 通常の推論
        if hasattr(self.model, 'model_state'):
            self.model.model_state = {}  # type: ignore

        outputs = self.model(x, **kwargs)

        # 2. 適応 (Adaptation)
        self._apply_adaptation()

        return outputs

    def _apply_adaptation(self):
        if not hasattr(self.model, 'model_state'):
            return

        model_state = getattr(self.model, 'model_state', {})
        original_lr = self.learning_rule.learning_rate
        self.learning_rule.learning_rate *= self.adaptation_rate_multiplier

        try:
            for name, module in self.model.named_modules():
                if self.target_layers and name not in self.target_layers:
                    continue

                pre = model_state.get(f"pre_activity_{name}")
                post = model_state.get(f"post_activity_{name}")

                if pre is not None and post is not None and hasattr(module, 'weight'):
                    dw, _ = self.learning_rule.update(pre, post, module.weight)
                    with torch.no_grad():
                        module.weight += dw
                        module.weight.clamp_(min=-1.0, max=1.0)
        finally:
            self.learning_rule.learning_rate = original_lr


class FastAdaptationTrainer:
    optimizer: Optional[torch.optim.Optimizer]

    def __init__(self, model: nn.Module, plasticity_layers: List[str], optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam, lr: float = 1e-3):
        self.model = model
        self.plasticity_layers = plasticity_layers
        self.trainable_params: List[nn.Parameter] = []
        self._configure_plasticity()

        if self.trainable_params:
            self.optimizer = optimizer_cls(
                self.trainable_params, lr=lr)  # type: ignore[call-arg]
        else:
            self.optimizer = None

    def _configure_plasticity(self):
        for name, param in self.model.named_parameters():
            is_plastic = any(
                layer_name in name for layer_name in self.plasticity_layers)
            param.requires_grad = is_plastic
            if is_plastic:
                self.trainable_params.append(param)

    def adapt_on_batch(self, inputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> float:
        if not self.optimizer:
            return 0.0
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = criterion(logits, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
