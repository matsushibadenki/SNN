# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: 対照学習・サロゲート勾配版)

import torch
import torch.nn as nn
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        
        # 閾値を低めに設定し、初期の発火を促す
        self.threshold = max_states * 0.2
        
        # --- 初期化の改善 ---
        # Forward-Forwardの考えに基づき、初期はランダムな射影として機能させる
        # 平均を閾値付近に設定し、分散を持たせることで多様な特徴を抽出
        states = torch.randn(out_features, in_features) * 20.0 + self.threshold
        self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 2.0))
        self.register_buffer('refractory_count', torch.zeros(out_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 重み: 閾値を超えたら1、負の閾値を下回ったら-1、それ以外0 (1.58bit)
        # または単純なBinary (0, 1)ゲートとして動作
        w = torch.zeros_like(self.states)
        w[self.states > self.threshold] = 1.0
        w[self.states < -self.threshold] = -1.0 # 抑制性結合も許可
        return w

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 入力がスパースでも反応するようにゲインを調整
        # Moduloタスクのように特定のビットの組み合わせが重要な場合、感度を上げる
        current = torch.matmul(x, w.t())
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        is_refractory = (ref_count > 0).float().unsqueeze(0)
        effective_current = current * (1.0 - is_refractory)
        
        # 膜電位の更新 (リークあり)
        new_v_mem = v_mem.unsqueeze(0) * 0.5 + effective_current
        
        # 発火判定
        spikes = (new_v_mem >= v_th.unsqueeze(0)).float()
        
        # 状態更新
        mean_spikes = spikes.mean(dim=0)
        new_refractory = (ref_count - 1.0).clamp(0) + mean_spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        # 発火したニューロンの電位をリセット（ソフトリセット）
        v_mem_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
        self.membrane_potential.copy_(v_mem_next)
        
        # 閾値の恒常性維持 (Homeostasis)
        with torch.no_grad():
            target_activity = 0.15
            th_update = (mean_spikes - target_activity) * 0.05
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(1.0, 20.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        """
        Forward-Forward / Feedback Alignmentに基づく可塑性更新
        reward: この層に対する「誤差信号」または「適合度信号」として扱う
        """
        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            # rewardがスカラーの場合はテンソル化
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # サロゲート勾配の近似
            # 発火(post_spikes)の有無に関わらず、膜電位が閾値に近いほど更新を受け入れやすくする
            # ここでは簡易的に、発火したニューロンとその周辺を活性化とみなす
            # LogicGatedなので、単純なHebbian (Pre * Post * Reward) をベースにする
            
            # Delta Rule: ΔW = lr * Error * Input
            # ここで reward を Error (Target - Output) とみなす
            
            # 学習率: 熟練度が低いほど高く設定
            lr = 5.0 * (1.0 - self.proficiency.item() * 0.5)
            
            # Error Signal (Reward) が (Batch, Out)
            # Pre Spikes が (Batch, In)
            # Delta W = (Out, In)
            
            # 勾配の計算: Feedback Alignment
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            # 重みの更新
            self.states.add_(delta * lr)
            
            # 重みの減衰 (Weight Decay) - 不要な接続を剪定
            self.states.mul_(0.9995)
            
            # ノイズ注入 (確率的探索)
            noise = torch.randn_like(self.states) * 0.1
            self.states.add_(noise)
            
            # クランプ
            self.states.clamp_(-self.max_states, self.max_states)
            
            # 熟練度の更新
            reward_mean = reward.abs().mean().item()
            # 誤差が小さい（報酬が0に近い）場合は熟練度が上がったとみなす... というよりは
            # ここでは単純に学習回数としてカウント
            self.proficiency.add_(0.0001)
            self.proficiency.clamp_(0.0, 1.0)
