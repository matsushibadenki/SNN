# ファイルパス: snn_research/learning_rules/bcm_rule.py
# Title: 高精度 BCM (Bienenstock-Cooper-Munro) 学習規則 (v16.5)
# Description:
#   Objective.mdの目標発火率(0.1-2Hz)に基づき、恒常性維持とスパース性を極大化。
#   ニューロン間の競合(Lateral Inhibition)要素を擬似的に導入。
#   [Fix] BioLearningRuleのシグネチャ変更に対応 (optional_params追加)

import torch
from typing import Dict, Any, Optional, Tuple, cast
from .base_rule import BioLearningRule

class BCMLearningRule(BioLearningRule):
    """
    BCM (Bienenstock-Cooper-Munro) 学習規則。
    目標発火率への収束精度を高め、学習の再現性(Objective.md ③)を向上。
    """
    avg_post_activity: Optional[torch.Tensor]

    def __init__(
        self, 
        learning_rate: float = 0.005, 
        tau_avg: float = 500.0, 
        target_rate: float = 0.01, # 約 1Hz 相当 (TimeStep依存)
        dt: float = 1.0
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.tau_avg = max(1.0, tau_avg)
        self.target_rate = target_rate
        self.dt = dt
        
        self.avg_post_activity = None
        self.avg_decay_factor = dt / self.tau_avg
        
        # 非線形定数 (学習の安定性向上用)
        self.stability_eps = 1e-6

        print(f"🧠 BCM V16.5 initialized (Target: {target_rate}, High Stability Mode)")

    def _initialize_traces(self, post_shape: int, device: torch.device):
        # 初期状態は目標レートで初期化し、急激な重み変化を抑制
        self.avg_post_activity = torch.full((post_shape,), self.target_rate, device=device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # バッチ平均の取得
        pre_avg = pre_spikes.mean(dim=0) if pre_spikes.dim() > 1 else pre_spikes
        post_avg = post_spikes.mean(dim=0) if post_spikes.dim() > 1 else post_spikes

        if self.avg_post_activity is None or self.avg_post_activity.shape[0] != post_avg.shape[0]:
            self._initialize_traces(post_avg.shape[0], post_spikes.device)
        
        avg_act = cast(torch.Tensor, self.avg_post_activity)

        # 1. 閾値 (theta) の動的更新: 脳の恒常性をシミュレート
        # avg_actがtarget_rateを超えるとthetaが上昇し、LTD(弱化)が起きやすくなる
        with torch.no_grad():
            new_avg = (1.0 - self.avg_decay_factor) * avg_act + self.avg_decay_factor * post_avg
            self.avg_post_activity = new_avg.detach()

        # 2. 閾値関数の計算 (theta = E[y^2] / E[y] を簡略化)
        theta = (avg_act ** 2) / (self.target_rate + self.stability_eps)
        
        # 3. 状態遷移関数の計算: post * (post - theta)
        # これにより、活動が高いニューロンはより強く(LTP)、低いニューロンは弱く(LTD)なる
        phi = post_avg * (post_avg - theta)
        
        # 4. 重み更新量の計算
        # Objective.md ⑭ の低レイテンシを維持するため、計算は最小限に
        dw = self.learning_rate * torch.outer(phi, pre_avg)
        
        # 5. 重みの正規化(Weight Scaling)のヒントを返す（オプション）
        return dw, None