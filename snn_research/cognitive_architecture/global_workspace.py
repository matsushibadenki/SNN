# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# 日本語タイトル: Global Workspace (Consciousness Hub) v1.0
# 目的・内容:
#   ROADMAP Phase 4 "Consciousness" の中核。
#   複数の無意識モジュール（Specialist Modules）からの入力を集約し、
#   注意機構（Attention）によって最も重要な情報を選択（着火）し、
#   それを脳全体にブロードキャストすることで「意識」を形成する。

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class GlobalWorkspace(nn.Module):
    """
    グローバル・ワークスペース（GWT）。
    脳内の情報の「競合」と「放送」を管理する。
    """

    def __init__(
        self,
        dim: int = 64,
        num_slots: int = 1,  # 一度に意識できる事象の数（通常は1〜数個）
        decay: float = 0.9  # 意識の持続性（ワーキングメモリ的性質）
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.decay = decay

        # 意識の内容（Global Working Memory）
        self.register_buffer("workspace_state", torch.zeros(1, dim))

        # Attention Mechanism (Selector)
        # 各モジュールからの入力の「重要度」を評価する
        self.selector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)  # Importance Score
        )

        logger.info("👁️ Global Workspace (Consciousness) initialized.")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Args:
            inputs: 各モジュールからの入力辞書
                    {"vision": tensor_v, "pain": tensor_p, "thought": tensor_t, ...}
                    各Tensorは [Batch, Dim] または [Batch, Seq, Dim]

        Returns:
            broadcast: 意識に選ばれた情報（全モジュールへ送信される）
            winner: 選ばれたモジュールの名前
        """
        # 1. 入力の前処理（次元統一）
        candidates = []
        names = []

        for name, tensor in inputs.items():
            # 平均プーリング等で [Batch, Dim] に揃える
            if tensor.dim() > 2:
                flat_tensor = tensor.mean(dim=1)
            else:
                flat_tensor = tensor

            # 次元チェック（異なる場合は射影が必要だが、ここでは同一と仮定するかゼロ埋め）
            if flat_tensor.shape[-1] != self.dim:
                # 簡易リサイズ（実運用では専用のAdapterが必要）
                if flat_tensor.shape[-1] < self.dim:
                    pad = self.dim - flat_tensor.shape[-1]
                    flat_tensor = F.pad(flat_tensor, (0, pad))
                else:
                    flat_tensor = flat_tensor[:, :self.dim]

            candidates.append(flat_tensor)
            names.append(name)

        if not candidates:
            return {"broadcast": self.workspace_state, "winner": None}

        # スタック: [Num_Modules, Batch, Dim]
        # Batch=1前提で簡易化: [Num_Modules, Dim]
        stack = torch.cat(candidates, dim=0)

        # 2. 競合 (Competition) - Bottom-up Attention
        # 各情報の「Salience（顕著性）」を計算
        scores = self.selector(stack).squeeze(-1)  # [Num_Modules]

        # Softmaxで確率分布にする（確率的選択も可能だが、ここではWinner-Take-All）
        # ノイズを加えて揺らぎを持たせる（カオス的遍歴）
        noise = torch.randn_like(scores) * 0.1
        probs = F.softmax(scores + noise, dim=0)

        # 最も強い信号を選択
        winner_idx = torch.argmax(probs).item()
        winner_name = names[winner_idx]
        winner_content = candidates[winner_idx]

        # 3. 放送 (Broadcast) - Update Global State
        # 前回の意識状態とブレンド（思考の流れ）
        new_state = (1 - self.decay) * winner_content + \
            self.decay * self.workspace_state

        # Update buffer
        # Detach to prevent infinite graph history
        self.workspace_state = new_state.detach()

        return {
            "broadcast": new_state,     # 全脳へ送信される信号
            "winner": winner_name,      # 意識に上がったモジュール名
            "salience": probs.detach()  # 各モジュールの注目度
        }

    def get_current_thought(self):
        return self.workspace_state
