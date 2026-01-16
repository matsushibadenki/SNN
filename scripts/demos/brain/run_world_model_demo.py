# ファイルパス: scripts/demos/brain/run_world_model_demo.py
# Title: Meta-Cognition & World Model Simulation Demo (Final Fix v6)
# Description:
#   System 1 (直感) と System 2 (熟考/シミュレーション) の切り替えデモ。
#   修正 v6: SNNCoreの初期化パラメータ(in_features, out_features, architecture_type)を
#           正しく設定し、Dimension mismatch警告を解消。

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.models.experimental.world_model_snn import SpikingWorldModel
from snn_research.core.snn_core import SNNCore  # Explicit import for override

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("SWM_Demo")

class DemoWorldModelWrapper(SpikingWorldModel):
    """
    デモ用にSpikingWorldModelを拡張したラッパークラス。
    不足している encode メソッドや simulate_trajectory メソッドを補完します。
    """
    def __init__(self, **kwargs):
        # 必須引数のデフォルト値を設定
        defaults = {
            'vocab_size': 0,
            'd_state': 64,
            'num_layers': 2,
            'time_steps': 16,
            'sensory_configs': {'vision': 64},
            'neuron_config': {'type': 'LIF', 'v_th': 0.5, 'beta': 0.9}
        }
        # kwargsで指定された値があればそれを優先
        for k, v in kwargs.items():
            defaults[k] = v
            
        super().__init__(
            vocab_size=defaults['vocab_size'],
            action_dim=defaults['action_dim'],
            d_model=defaults['d_model'],
            d_state=defaults['d_state'],
            num_layers=defaults['num_layers'],
            time_steps=defaults['time_steps'],
            sensory_configs=defaults['sensory_configs'],
            neuron_config=defaults['neuron_config']
        )
        
        # [Fix v6] transition_modelの初期化設定をSNNCoreの仕様に完全適合させる
        # - architecture_type: 正しいキー名に変更
        # - in_features/out_features: d_modelに一致させ、フォールバック時も次元を保証
        self.transition_model = SNNCore(
            config={
                "d_model": defaults['d_model'],
                "in_features": defaults['d_model'],   # Fallback用: 入力次元
                "hidden_features": defaults['d_model']*2,
                "out_features": defaults['d_model'],  # Fallback用: 出力次元 (重要)
                "num_layers": defaults['num_layers'],
                "time_steps": defaults['time_steps'],
                "neuron": defaults['neuron_config'],
                "architecture_type": "spiking_mamba"  # キー名を修正
            },
            vocab_size=defaults['d_model']
        )
        
        # 報酬予測用の簡易ヘッドを追加（デモ用）
        self.reward_predictor = nn.Linear(defaults['d_model'], 1)

    def to(self, device):
        """
        PyTorchの .to() は再帰的にパラメータを移動させるが、
        サブモジュールのカスタム .to() (self.deviceの更新など) は呼ばないため、ここで明示的に呼ぶ。
        """
        super().to(device)
        
        # UniversalSpikeEncoder の device 変数を更新
        if hasattr(self.encoder, 'to'):
            self.encoder.to(device)
            
        # SNNCore の device 変数も更新
        if hasattr(self.transition_model, 'to'):
            self.transition_model.to(device)
            # SNNCore内部でdeviceプロパティを持っていれば更新 (snn_core.pyの実装による)
            if hasattr(self.transition_model, 'device'):
                self.transition_model.device = torch.device(device)
            
        return self

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """
        現在の観測を潜在状態にエンコードする。
        observation: (Batch, InputDim) -> (1, 64)
        """
        # UniversalSpikeEncoder._encode_image は (B, Dim) を受け取ると
        # 自動的に (B, Time, Dim) にレートコーディングで拡張します。
        
        sensory_spikes = {
            'vision': self.encoder.encode(observation, modality='vision')
        }
        
        # Projectorに通して潜在状態 z_t を得る
        z_t = self.projector(sensory_spikes)
        return z_t

    def simulate_trajectory(self, initial_latent: torch.Tensor, actions: torch.Tensor) -> dict:
        """
        潜在状態から行動系列に基づいて未来をシミュレーションする。
        initial_latent: (Batch, Time, D_model)
        actions: (Batch, Steps, ActionDim)
        """
        batch_size, steps, _ = actions.shape
        
        # 1. 行動のエンコード
        a_t = self.action_encoder(actions) # (B, Steps, D_model)
        
        # 時間次元の整合性調整
        # initial_latentは (B, T_enc, D) なので、シミュレーション用に調整
        # ここでは単純化のため、initial_latentの平均を初期状態として使う
        curr_state = initial_latent.mean(dim=1, keepdim=True).repeat(1, steps, 1) # (B, Steps, D)
        
        # 2. 状態遷移 (Transition)
        # 入力 = 現在の状態 + 行動
        transition_input = curr_state + a_t
        
        # Transition Modelを実行 (Mamba/SNN Core)
        transition_out = self.transition_model(transition_input)
        
        if isinstance(transition_out, tuple):
            z_next_pred = transition_out[0]
        else:
            z_next_pred = transition_out
        
        # [Safety] 次元チェック (Fix v6によりここは通過するはずだが、念のため残す)
        if z_next_pred.shape[-1] != self.d_model:
            logger.warning(f"⚠️ Dimension mismatch in transition output: {z_next_pred.shape}, expected {self.d_model}. Using fallback projection.")
            if z_next_pred.shape[-1] < self.d_model:
                z_next_pred = F.pad(z_next_pred, (0, self.d_model - z_next_pred.shape[-1]))
            else:
                z_next_pred = z_next_pred[..., :self.d_model]

        # 3. 報酬予測 (デモ用)
        # 予測された潜在状態から報酬を計算
        rewards = self.reward_predictor(z_next_pred) # (B, Steps, 1)
        
        return {
            "predicted_states": z_next_pred,
            "rewards": rewards
        }

def main():
    print("🌍 --- Spiking World Model & Meta-Cognition Demo ---")

    # デバイス設定 (MPS/CUDA/CPU)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    
    print(f"🖥️  Using device: {device}")
    
    # 1. コンポーネントの初期化 & デバイス転送
    # System 1 Monitor
    meta_snn = MetaCognitiveSNN(d_model=10).to(device)
    
    # System 2 Simulator (World Model)
    # ラッパークラスを使用し、必要な引数を渡す
    swm = DemoWorldModelWrapper(
        action_dim=4, 
        d_model=128,
        # input_dim=64 は sensory_configs={'vision': 64} として内部で処理されます
    ).to(device)
    
    # 2. System 1 のシミュレーション (不確実性の検知)
    print("\n🔹 Phase 1: Meta-Cognitive Monitoring")
    
    # ケースA: 自信がある (Low Entropy)
    logits_confident = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]], device=device)
    res_a = meta_snn.monitor_system1_output(logits_confident)
    print(f"   Case A (Confident): Entropy={res_a['entropy']:.4f} -> Trigger System 2? {res_a['trigger_system2']}")
    
    # ケースB: 迷っている (High Entropy)
    logits_uncertain = torch.tensor([[2.0, 2.0, 2.1, 1.9, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]], device=device)
    res_b = meta_snn.monitor_system1_output(logits_uncertain)
    print(f"   Case B (Uncertain): Entropy={res_b['entropy']:.4f} -> Trigger System 2? {res_b['trigger_system2']}")
    
    if res_b['trigger_system2']:
        print("   🚨 Uncertainty detected! Switching to Deep Thought Mode (System 2)...")
        
    # 3. 脳内シミュレーション (World Modelによる計画)
    print("\n🔹 Phase 2: Mental Simulation (Planning)")
    print("   Simulating 3 different action plans to resolve uncertainty...")
    
    # 現在の観測 (Dummy Input) -> デバイス指定
    current_obs = torch.randn(1, 64, device=device)
    
    # ラッパーに追加した encode メソッドを使用
    initial_latent = swm.encode(current_obs)
    
    # 3つの行動プラン (Action Sequences)
    # (Batch, Steps, ActionDim) -> デバイス指定
    # Plan 1: ランダムな行動
    plan_1 = torch.randn(1, 5, 4, device=device) 
    # Plan 2: 待機 (全て0.0に近い)
    plan_2 = torch.zeros(1, 5, 4, device=device) + 0.1
    # Plan 3: 特定の行動 (右へ行くなど)
    plan_3 = torch.zeros(1, 5, 4, device=device)
    plan_3[:, :, 0] = 2.0 # Strong action on dim 0
    
    plans = {"Random": plan_1, "Wait": plan_2, "Act": plan_3}
    best_plan = None
    max_reward = -float('inf')
    
    for name, actions in plans.items():
        # ラッパーに追加した simulate_trajectory メソッドを使用
        sim_result = swm.simulate_trajectory(initial_latent, actions)
        
        # 予測された累積報酬
        total_reward = sim_result["rewards"].sum().item()
        
        print(f"   - Plan '{name}': Expected Reward = {total_reward:.4f}")
        
        if total_reward > max_reward:
            max_reward = total_reward
            best_plan = name
            
    print(f"\n✅ Decision: Selected Plan '{best_plan}' based on mental simulation.")
    
    # 4. フィードバックと学習 (Surprise Detection)
    print("\n🔹 Phase 3: Reality Check (Surprise)")
    
    predicted_next = initial_latent 
    # 実際の結果が予測と大きく異なるとする (Surprise!)
    actual_next = initial_latent + torch.randn_like(initial_latent) * 3.0 
    
    surprise = meta_snn.evaluate_surprise(predicted_next, actual_next)
    print(f"   Surprise Value: {surprise:.4f}")
    
    if surprise > 0.5:
        print("   🧠 Brain detects high surprise! Updating World Model weights (Simulated).")

    print("\n🎉 Demo Completed.")

if __name__ == "__main__":
    main()