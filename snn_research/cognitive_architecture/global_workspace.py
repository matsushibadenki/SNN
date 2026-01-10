# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# 日本語タイトル: Global Workspace (Consciousness Hub) v1.2
# 修正内容: DIコンテナからの model_registry 注入に対応し、テスト用の get_information を追加。

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Callable

logger = logging.getLogger(__name__)


class GlobalWorkspace(nn.Module):
    """
    グローバル・ワークスペース（GWT）。
    脳内の情報の「競合」と「放送」を管理する。
    """
    workspace_state: torch.Tensor

    def __init__(
        self,
        dim: int = 64,
        num_slots: int = 1,
        decay: float = 0.9,
        model_registry: Optional[Any] = None  # Added for DI compatibility
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.decay = decay
        self.model_registry = model_registry

        # 意識の内容（Global Working Memory）
        self.register_buffer("workspace_state", torch.zeros(1, dim))

        # Attention Mechanism (Selector)
        self.selector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

        # Subscribers
        self.subscribers: List[Callable[[str, Any], None]] = []
        self.current_content: Dict[str, Any] = {}

        logger.info("👁️ Global Workspace (Consciousness) initialized.")

    def subscribe(self, callback: Callable[[str, Any], None]):
        """他のモジュールが意識の放送を受信するために登録する"""
        self.subscribers.append(callback)

    def upload_to_workspace(self, source: str, data: Any, salience: float = 0.5):
        """モジュールからワークスペースへの情報提供"""
        if salience > 0.7:
            if isinstance(data, dict) and "vector_state" in data:
                vec = data["vector_state"]
                if isinstance(vec, torch.Tensor):
                    if vec.shape[-1] == self.dim:
                        self.workspace_state = vec.detach()

            self._broadcast_to_subscribers(source, data)

    def broadcast(self, inputs: List[Any], context: Optional[str] = None) -> Any:
        """Legacy Interface"""
        tensor_inputs = {}
        for i, item in enumerate(inputs):
            if isinstance(item, torch.Tensor):
                tensor_inputs[f"input_{i}"] = item
            elif isinstance(item, dict) and "features" in item:
                tensor_inputs[f"module_{i}"] = item["features"]

        if tensor_inputs:
            result = self.forward(tensor_inputs)
            self._broadcast_to_subscribers(
                str(result["winner"]), result["broadcast"])
            return result["broadcast"]

        return self.workspace_state

    def get_current_thought(self) -> torch.Tensor:
        return self.workspace_state

    def get_information(self) -> torch.Tensor:
        """Alias for test compatibility (test_cognitive_components.py)"""
        return self.get_current_thought()

    def get_current_content(self) -> Dict[str, Any]:
        """
        [Phase 3.1] 現在の意識内容（辞書形式）を取得する。
        ExplainabilityEngine等で使用。
        """
        return self.current_content

    def _broadcast_to_subscribers(self, source: str, content: Any):
        # コンテンツの保持
        if isinstance(content, dict):
            self.current_content = content
        else:
            self.current_content = {"type": "raw",
                                    "data": content, "source": source}

        for callback in self.subscribers:
            try:
                callback(source, content)
            except Exception as e:
                logger.warning(f"Broadcast error: {e}")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        candidates = []
        names = []

        for name, tensor in inputs.items():
            if tensor.dim() > 2:
                flat_tensor = tensor.mean(dim=1)
            else:
                flat_tensor = tensor

            if flat_tensor.shape[-1] != self.dim:
                if flat_tensor.shape[-1] < self.dim:
                    pad = self.dim - flat_tensor.shape[-1]
                    flat_tensor = F.pad(flat_tensor, (0, pad))
                else:
                    flat_tensor = flat_tensor[:, :self.dim]

            candidates.append(flat_tensor)
            names.append(name)

        if not candidates:
            return {"broadcast": self.workspace_state, "winner": None}

        stack = torch.cat(candidates, dim=0)
        scores = self.selector(stack).squeeze(-1)
        noise = torch.randn_like(scores) * 0.1
        probs = F.softmax(scores + noise, dim=0)

        winner_idx = int(torch.argmax(probs).item())
        winner_name = names[winner_idx]
        winner_content = candidates[winner_idx]

        new_state = (1 - self.decay) * winner_content + \
            self.decay * self.workspace_state
        self.workspace_state = new_state.detach()

        return {
            "broadcast": new_state,
            "winner": winner_name,
            "salience": probs.detach()
        }
