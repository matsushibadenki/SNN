# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# Title: RL Agent (Mypy Fix)
# 修正: experience_bufferへのappend時の型エラーを修正 (Optional型の除外)。

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Any, List, Optional, Type, cast

from snn_research.core.hybrid_core import HybridNeuromorphicCore

# Transformerモデルのインポート試行
SpikingDSATransformerCls: Optional[Type[nn.Module]] = None
try:
    from snn_research.models.transformer.dsa_transformer import SpikingDSATransformer
    SpikingDSATransformerCls = SpikingDSATransformer
except ImportError:
    pass


class ReinforcementLearnerAgent:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        device: str,
        model_config: Optional[Dict[str, Any]] = None,
        synaptic_rule: Optional[Any] = None,
        homeostatic_rule: Optional[Any] = None
    ):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size

        self.synaptic_rule = synaptic_rule
        self.homeostatic_rule = homeostatic_rule

        self.use_transformer = False
        self.model: nn.Module
        self.optimizer: Optional[optim.Optimizer] = None

        if model_config and model_config.get('architecture') == 'dsa_transformer':
            if SpikingDSATransformerCls is None:
                raise ImportError("SpikingDSATransformer not found.")

            print("  🧠 [Agent] Initializing Scaled SNN-DSA Transformer Brain...")
            self.model = SpikingDSATransformerCls(
                input_dim=input_size,
                d_model=model_config.get('d_model', 512),
                num_heads=model_config.get('num_heads', 8),
                num_layers=model_config.get('num_layers', 6),
                dim_feedforward=model_config.get('dim_feedforward', 2048),
                time_window=16,
                num_classes=output_size,
                use_bitnet=True
            ).to(device)
            self.use_transformer = True
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=0.0001, weight_decay=0.01)

        else:
            hidden_dim = 64
            self.model = HybridNeuromorphicCore(
                in_features=input_size,
                hidden_features=hidden_dim,
                out_features=output_size
            ).to(device)
            self.optimizer = None

        self.experience_buffer: List[Dict[str, torch.Tensor]] = []

        self.base_lr = 0.05
        self.min_lr = 0.001
        self.lr_decay = 0.998
        self.current_lr = self.base_lr

        self.best_model_state: Optional[Dict[str, Any]] = None
        self.best_reward_avg = -float('inf')
        self.current_reward_avg = 0.0
        self.alpha_avg = 0.2
        self.rollback_threshold = 0.3

    def get_action(self, state: torch.Tensor, record_experience: bool = True) -> int:
        self.model.eval()
        if record_experience:
            self.model.train()

        pre_activity: Optional[torch.Tensor] = None

        with torch.no_grad():
            if state.dim() == 1:
                state_input = state.unsqueeze(0)
            else:
                state_input = state

            if self.use_transformer:
                logits = self.model(state_input)
                if isinstance(logits, tuple):
                    logits = logits[0]
            else:
                hybrid_model = cast(HybridNeuromorphicCore, self.model)
                f = hybrid_model.fast_process(state_input)
                r = hybrid_model.deep_process(f)
                out_result = hybrid_model.output_gate(r)
                logits = out_result['output'] if isinstance(
                    out_result, dict) else out_result
                
                pre_activity = r.clone()

            if self.model.training:
                decay_progress = max(
                    0.0, (self.current_lr - self.min_lr) / (self.base_lr - self.min_lr))
                temperature = 0.1 + 1.5 * (decay_progress ** 2)
            else:
                temperature = 0.1

            probs = torch.softmax(logits / temperature, dim=1)
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / self.output_size

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            action_idx = int(action.item())

            if record_experience:
                # [Mypy Fix] pre_activityがNoneの場合のデフォルト値を設定するか、
                # 辞書構築時に条件分岐して型安全にする
                step_data: Dict[str, torch.Tensor] = {
                    'state': state_input.clone(),
                    'action': torch.tensor(action_idx),
                    'probs': probs.clone(),
                }
                
                if pre_activity is not None:
                    step_data['pre_spikes'] = pre_activity
                else:
                    # Hybridでない場合など、ダミーを入れるかキーを含めない
                    # ここでは型合わせのためダミーを入れる
                    step_data['pre_spikes'] = torch.empty(0) 

                self.experience_buffer.append(step_data)

            return action_idx

    def learn(self, reward: float) -> None:
        if not self.experience_buffer:
            return

        current_step = self.experience_buffer[-1]
        trajectory = {
            'spikes_history': [current_step],
            'total_reward': reward
        }

        self.learn_with_grpo([trajectory])
        self.experience_buffer.clear()

    def _save_checkpoint(self, current_avg: float):
        self.best_reward_avg = current_avg
        self.best_model_state = copy.deepcopy(
            {k: v.cpu() for k, v in self.model.state_dict().items()})

    def _restore_checkpoint(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)
            self.current_reward_avg = self.best_reward_avg * 0.90
            self.current_lr = max(self.min_lr, self.current_lr * 0.8)
            if self.optimizer:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr

    def learn_with_grpo(self, trajectories: List[Dict[str, Any]], baseline_reward: float = 0.0):
        if not trajectories:
            return

        self.model.train()

        rewards_list = [t['total_reward'] for t in trajectories]
        total_rewards = torch.tensor(
            rewards_list, dtype=torch.float32, device=self.device)
        batch_mean_reward = total_rewards.mean().item()

        if self.current_reward_avg == 0.0:
            self.current_reward_avg = batch_mean_reward
        else:
            self.current_reward_avg = (
                1 - self.alpha_avg) * self.current_reward_avg + self.alpha_avg * batch_mean_reward

        if self.current_reward_avg > self.best_reward_avg and self.current_reward_avg > 0.15:
            self._save_checkpoint(self.current_reward_avg)
        elif self.best_reward_avg > 0.25 and self.current_reward_avg < self.best_reward_avg * (1 - self.rollback_threshold):
            self._restore_checkpoint()
            return

        if len(total_rewards) > 1:
            mean_reward = total_rewards.mean()
            std_reward = total_rewards.std() + 1e-8
            advantages = (total_rewards - mean_reward) / std_reward
        else:
            advantages = torch.zeros_like(total_rewards)

        self.current_lr = max(self.min_lr, self.current_lr * self.lr_decay)

        if self.use_transformer and self.optimizer is not None:
            self.optimizer.zero_grad()
            loss = torch.tensor(0.0, device=self.device)

            for i, trajectory in enumerate(trajectories):
                adv = float(advantages[i].item())
                adv_weight = min(adv, 2.0) if adv > 0 else max(adv, -1.0)

                episode_history = trajectory.get('spikes_history', [])
                if not episode_history:
                    continue

                states = torch.cat([step['state']
                                   for step in episode_history], dim=0)
                actions = torch.tensor(
                    [step['action'] for step in episode_history], device=self.device)

                output = self.model(states)
                logits = output[0] if isinstance(output, tuple) else output

                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                selected_log_probs = log_probs.gather(
                    1, actions.unsqueeze(1)).squeeze(1)
                loss = loss - (selected_log_probs * adv_weight).mean()

            loss = loss / len(trajectories)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
            self.optimizer.step()

        else:
            hybrid_model = cast(HybridNeuromorphicCore, self.model)

            for i, trajectory in enumerate(trajectories):
                if hasattr(hybrid_model, 'reset_state'):
                    hybrid_model.reset_state()

                adv = float(advantages[i].item())
                clipped_reward = float(np.clip(adv, 0.0, 2.0)) if adv > 0 else float(
                    np.clip(adv, -1.0, 0.0))

                episode_history = trajectory.get('spikes_history', [])
                for step_data in episode_history:
                    pre_spikes = step_data.get('pre_spikes')
                    
                    if pre_spikes is None or pre_spikes.numel() == 0:
                        continue

                    action_idx = step_data['action']
                    target = torch.tensor(
                        [action_idx], device=self.device, dtype=torch.long)
                    dummy_post = {'output': torch.zeros(1, device=self.device)}
                    effective_lr = self.current_lr * clipped_reward

                    if hasattr(hybrid_model, 'output_gate'):
                        hybrid_model.output_gate.update_plasticity(
                            pre_activity=pre_spikes,
                            post_output=dummy_post,
                            target=target,
                            learning_rate=effective_lr
                        )