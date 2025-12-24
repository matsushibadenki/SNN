# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (ベクトル教示対応版)
# 目的: ベクトル報酬を受け取り、ニューロンごとに個別の学習を行うことで分類精度を劇的に向上させる。

import torch
import torch.nn as nn
from typing import cast, Union

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        # 閾値を少し下げて初期反応を良くする
        self.threshold = max_states // 2
        
        # 初期状態: 分散を大きくして多様な特徴を持たせる
        states = torch.randn(out_features, in_features) * 20.0 + (self.threshold - 5.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        # アダプティブ閾値の初期値を適切に設定
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 10.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))
        self.register_buffer('refractory_count', torch.zeros(out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 重みは 0 or 1 (バイナリ接続)
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # ゲイン調整: 接続数が多いニューロンほど感度を下げる（正規化のような役割）
        conn_count = w.sum(dim=1).clamp(min=1.0)
        # 以前よりも少しゲインを高めに維持し、信号が伝わりやすくする
        gain = 15.0 / torch.log1p(conn_count * 0.5)
        
        current = torch.matmul(x, w.t()).view(-1) * gain
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        is_refractory = (ref_count > 0).float()
        effective_current = current * (1.0 - is_refractory)
        
        # ノイズ注入: デッドロック回避
        noise = torch.randn_like(v_mem) * 0.5
        v_mem.mul_(0.8).add_(effective_current + noise) # 膜電位のリークを早める(0.9->0.8)ことでノイズ耐性を向上
        
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 不応期
        new_refractory = (ref_count - 1.0).clamp(0) + spikes * 3.0 # 発火したら3ステップ休む
        self.refractory_count.copy_(new_refractory)
        
        with torch.no_grad():
            # 適格性トレース（Eligibility Trace）の更新
            # Preが発火し、かつPostも発火しそうな状態（膜電位が高い）を記録する
            # x.view(-1) は入力ベクトル
            # v_mem > v_th * 0.5 のような「発火しそうだった」情報も少し加味すると学習が早い
            
            # シンプルなヘブ則トレース: Pre * Post
            # ただしここでは報酬が遅延して来るため、Preの発火履歴を残す
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1))) # Post発火時のみ更新だと強すぎるので修正
            # Preのみの履歴も保持すべきだが、メモリ節約のためこの実装では
            # 「Postが発火した瞬間のPreの状態」を強く学習する方式をとる
            
            # トレースの減衰とクリップ
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 閾値のホメオスタシス
            # 発火したら閾値を上げ、発火しないなら下げる
            target_activity = 0.1 # 目標発火率
            
            # 発火過多なら閾値を上げ、過少なら下げる
            th_update = (spikes - target_activity) * 0.5
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(5.0, 40.0) # 下限と上限を設定
        
        # ハードリセット: 発火したら電位を0にする
        v_mem_reset = v_mem * (1.0 - spikes)
        self.membrane_potential.copy_(v_mem_reset)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor] = 0.0) -> None:
        """ 
        ベクトル報酬に対応した可塑性更新則 
        rewardがTensorの場合、各出力ニューロンごとに異なる学習信号（正解への強化、不正解への抑制）が適用される。
        """
        with torch.no_grad():
            # 熟練度の更新（単純な平均）
            if isinstance(reward, torch.Tensor):
                avg_reward = reward.mean().item()
            else:
                avg_reward = reward
                
            is_success = 1.0 if avg_reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # Modulationの計算
            if isinstance(reward, torch.Tensor):
                # ベクトル報酬: (out_features,) -> (out_features, 1) に変形してブロードキャスト
                modulation = reward.unsqueeze(1)
            else:
                modulation = reward
            
            # 学習率: 初心者は大きく、熟練者は小さく
            lr = 2.0 * (1.0 - prof * 0.8)
            
            # メインの重み更新 (Reward-Modulated Hebbian)
            # modulation > 0 (正解): トレース（Pre*Post）に基づいて強化
            # modulation < 0 (不正解): トレースに基づいて抑制（罰）
            
            # ここでのtraceは「Postが発火したときにPreがどうだったか」を記録している
            # 正解ニューロン(Modulation>0)でTraceが高い -> 正しい入力に反応した -> 強化
            # 不正解ニューロン(Modulation<0)でTraceが高い -> 間違った入力に反応した -> 抑制
            
            # 重み更新
            delta = trace * modulation * lr
            self.states.add_(delta)
            
            # 構造恒常性 (Structural Homeostasis)
            # 接続率を一定(20%程度)に保つように、全体的に値を底上げまたは抑制する
            current_conn = (self.states > self.threshold).float().mean(dim=1, keepdim=True)
            target_conn = 0.20
            conn_error = target_conn - current_conn
            
            # 接続が少なすぎるなら全体的に強化、多すぎるなら抑制
            self.states.add_(conn_error * 2.0)
            
            # 忘却とノイズ（過学習防止）
            self.states.mul_(0.9995) # 非常に緩やかな減衰
            self.states.add_(torch.randn_like(self.states) * 0.02)

            self.states.clamp_(1, self.max_states)
