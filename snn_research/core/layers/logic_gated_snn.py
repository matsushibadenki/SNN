# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: 次元維持＆ゲイン安定化版)

import torch
import torch.nn as nn
from typing import cast, Union

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.threshold = max_states // 2
        
        # 初期状態: 分散を大きくして多様な特徴を持たせる
        states = torch.randn(out_features, in_features) * 20.0 + (self.threshold - 5.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 10.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))
        self.register_buffer('refractory_count', torch.zeros(out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        # 入力: (batch, in_features) または (in_features,)
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        batch_size = x.size(0)
        
        # ゲイン調整
        # 修正: 分子を 15.0 -> 5.0 に下げて過活動（てんかん）を防止
        conn_count = w.sum(dim=1).clamp(min=1.0)
        gain = 5.0 / torch.log1p(conn_count * 0.5)
        
        # 電流計算
        # 修正: view(-1) を削除し、バッチ次元 (batch, out_features) を維持
        # w: (out, in), x: (batch, in) -> x @ w.t(): (batch, out)
        current = torch.matmul(x, w.t()) * gain.unsqueeze(0)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        # バッチ対応のためのブロードキャスト
        # v_mem, v_th, ref_count は (out_features,) なので (1, out_features) として扱う
        is_refractory = (ref_count > 0).float().unsqueeze(0)
        effective_current = current * (1.0 - is_refractory)
        
        noise = torch.randn_like(current) * 0.5
        
        # 膜電位更新 (In-place更新はバッチサイズ1を想定しているため、平均をとってバッファを更新する)
        # ※本来のSNNはバッチごとにステートを持つべきだが、この実装は「単一の脳」を想定しているため
        # バッチ内の平均的な活動で脳の状態（膜電位）を更新する簡易実装とする。
        new_v_mem = v_mem.unsqueeze(0) * 0.8 + effective_current + noise
        
        # 発火判定
        spikes = (new_v_mem >= v_th.unsqueeze(0)).to(torch.float32)
        
        # ステート更新 (バッチの平均または代表値で更新)
        # batch=1 前提のコードが多いが、汎用性のため平均を使用
        mean_spikes = spikes.mean(dim=0)
        
        new_refractory = (ref_count - 1.0).clamp(0) + mean_spikes * 3.0
        self.refractory_count.copy_(new_refractory)
        
        # 膜電位リセット (Hard reset)
        v_mem_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
        self.membrane_potential.copy_(v_mem_next)
        
        with torch.no_grad():
            # トレース更新 (Pre * Post)
            # x: (batch, in), spikes: (batch, out)
            # batch方向の平均をとって outer product 近似
            avg_x = x.mean(dim=0)
            self.eligibility_trace.mul_(0.9).add_(torch.outer(mean_spikes, avg_x))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 閾値ホメオスタシス
            target_activity = 0.1
            th_update = (mean_spikes - target_activity) * 0.2 # 感度を少し下げる(0.5->0.2)
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(5.0, 40.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor] = 0.0) -> None:
        """ 
        ベクトル報酬に対応した可塑性更新則 
        """
        with torch.no_grad():
            if isinstance(reward, torch.Tensor):
                avg_reward = reward.mean().item()
            else:
                avg_reward = reward
                
            is_success = 1.0 if avg_reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # Modulation
            if isinstance(reward, torch.Tensor) and reward.ndim == 1:
                modulation = reward.unsqueeze(1)
            else:
                modulation = reward
            
            lr = 2.0 * (1.0 - prof * 0.8)
            
            # 重み更新
            delta = trace * modulation * lr
            
            # 修正: 負の変調（罰）の場合、減少幅をクリップして「死」を防ぐ
            # modulation < 0 の要素だけ強さを0.2倍にする等の安全策
            if isinstance(modulation, torch.Tensor):
                # 罰の緩和: 負の部分だけ係数を掛けるのはTensor操作で複雑なので
                # 全体的に引きすぎないようにclampする
                delta = delta.clamp(min=-5.0, max=10.0) 
            
            self.states.add_(delta)
            
            # 構造恒常性
            current_conn = (self.states > self.threshold).float().mean(dim=1, keepdim=True)
            target_conn = 0.20
            conn_error = target_conn - current_conn
            
            # 接続回復力を強化
            self.states.add_(conn_error * 5.0)
            
            # 忘却とノイズ
            self.states.mul_(0.9995)
            self.states.add_(torch.randn_like(self.states) * 0.05) # ノイズを増やして再接続を促す

            self.states.clamp_(1, self.max_states)
