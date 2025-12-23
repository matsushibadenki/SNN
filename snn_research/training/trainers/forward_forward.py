# ファイルパス: snn_research/training/trainers/forward_forward.py
# 日本語タイトル: Forward-Forward アルゴリズムトレーナー (Ver 3.1 "Temporal Contrast")
# 機能説明:
#   1. Temporal Hard Negative: 時間構造を局所的に破壊し、SNNの時系列学習能力を強化。
#   2. Token Mixing: 文脈情報を攪乱する高度な偽データ生成。
#   3. Peer Normalization: バッチ内の競争原理による学習安定化。

from __future__ import annotations
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, Any, Optional, List, Union, cast

from snn_research.training.base_trainer import AbstractTrainer, DataLoader
from snn_research.core.networks.sequential_snn_network import SequentialSNNNetwork
from snn_research.core.network import AbstractNetwork

logger = logging.getLogger(__name__)

class ForwardForwardTrainer(AbstractTrainer):
    """
    Forward-Forward Algorithm (v3.1 Temporal Contrast)
    
    Quality Enhancements from Docs:
    - Temporal Hard Negative: 時系列データの「順序」や「タイミング」を破壊することで、
      SNNが時間的特徴（いつ発火するか）を学習するように強制する。
    - Robust Goodness: 数値安定性を高めたGoodness関数。
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.005,
        threshold: float = 2.0,
        logger_client: Optional[Any] = None
    ) -> None:
        super().__init__(model, logger_client)
        self.learning_rate = learning_rate
        self.threshold = threshold
        
        self.trainable_layers: List[nn.Module] = []
        self.layer_optimizers: List[torch.optim.Optimizer] = []
        self._setup_local_learning()

    def _setup_local_learning(self) -> None:
        """レイヤー探索とオプティマイザ設定"""
        self.trainable_layers = []
        self.layer_optimizers = []
        candidates: List[nn.Module] = []

        if isinstance(self.model, SequentialSNNNetwork):
            candidates = [self.model.layers_map[name] for name in self.model.layer_order]
        elif isinstance(self.model, AbstractNetwork):
            candidates = [cast(nn.Module, layer) for layer in self.model.layers]
        elif isinstance(self.model, nn.Module) and hasattr(self.model, 'layers') and isinstance(self.model.layers, nn.ModuleList):
            candidates = list(self.model.layers)
        elif isinstance(self.model, nn.Module):
            candidates = list(self.model.children())
        else:
            logger.warning(f"⚠️ FF Trainer: Unsupported model type {type(self.model)}.")
            return

        for layer in candidates:
            params = list(layer.parameters())
            if len(params) > 0:
                self.trainable_layers.append(layer)
                # AdamWに変更 (Weight Decayによる汎化性能向上)
                # ドキュメントにあるように、局所学習では学習率が重要
                optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=1e-4)
                self.layer_optimizers.append(optimizer)
        
        logger.info(f"🔍 FF Trainer: Configured {len(self.trainable_layers)} layers for Temporal Contrast training.")

    def _generate_hard_negative(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Hard Negative Mining (Enhanced):
        ドキュメントにある「Self-Contrastive」なアプローチを強化。
        バッチシャッフルに加え、時系列データの場合は時間構造を破壊する。
        """
        batch_size = inputs.size(0)
        neg_inputs = inputs.clone()
        device = inputs.device

        # Case A: テキスト/シーケンス (Batch, SeqLen) または (Batch, Time, Dim)
        # SNNへの入力は通常 (Batch, Time, ...) の形式を持つ
        is_sequence = False
        if inputs.dtype == torch.long: # Token IDs
            is_sequence = True
        elif inputs.dim() >= 3 and inputs.size(1) > 1: # Time dimension exists
            is_sequence = True

        if is_sequence:
            # 1. Batch Shuffle (Source Mixing)
            # 別のサンプルのデータを混ぜる（空間的/意味的な破壊）
            perm_indices = torch.randperm(batch_size, device=device)
            source_samples = inputs[perm_indices]
            
            mask_prob = 0.4 # 混合率を少し上げる
            mask = torch.rand(inputs.shape, device=device) < mask_prob
            neg_inputs[mask] = source_samples[mask]

            # 2. Temporal Shuffle (Time Mixing)
            # 同じサンプル内で、時間的な順序を部分的に入れ替える（時間的因果律の破壊）
            # SNNにとって「順序」は重要なので、これが強力なNegativeとなる
            if inputs.size(1) >= 4:
                # タイムステップのランダムな置換を行う
                time_steps = inputs.size(1)
                time_perm = torch.randperm(time_steps, device=device)
                
                # 全体をシャッフルすると簡単すぎるため、
                # バッチの半分に対してのみ時間シャッフルを適用するなどの工夫も有効だが、
                # ここでは「Hard」を目指して全体に適用
                
                # 注: 単純な転置だと次元が崩れるので注意して置換
                if inputs.dim() == 2: # (B, T)
                    neg_inputs_t = neg_inputs[:, time_perm]
                elif inputs.dim() == 3: # (B, T, D)
                    neg_inputs_t = neg_inputs[:, time_perm, :]
                elif inputs.dim() == 4: # (B, T, H, W) etc
                    neg_inputs_t = neg_inputs[:, time_perm, ...]
                else:
                    neg_inputs_t = neg_inputs
                
                # 時間シャッフルしたものと、バッチシャッフルしたものをランダムに採用
                mix_mask = torch.rand(batch_size, device=device) < 0.5
                final_neg = neg_inputs.clone()
                final_neg[mix_mask] = neg_inputs_t[mix_mask]
                return final_neg
            
            return neg_inputs

        # Case B: 画像 (Batch, C, H, W) - 時間次元なし
        elif inputs.dim() == 4:
            # Patch Shuffle的なチャネルシャッフル
            perm_indices = torch.randperm(batch_size, device=device)
            return inputs[perm_indices]

        # Case C: その他
        else:
            perm_indices = torch.randperm(batch_size, device=device)
            return inputs[perm_indices]

    def _calculate_temporal_goodness(self, h: torch.Tensor) -> torch.Tensor:
        """
        Temporal Goodness:
        SNNのスパイク活動の時間的なエネルギーを計算する。
        (Batch, Time, Dim) -> (Batch, )
        """
        if isinstance(h, tuple): h = h[0]
        if isinstance(h, dict): h = list(h.values())[0]

        # ドキュメントに基づき、エネルギーは二乗和（または平均）で定義
        # 時間次元がある場合、時間平均をとることで「持続的な活動」を評価する
        if h.dim() == 3: # (B, T, D)
            return h.pow(2).mean(dim=1).mean(dim=1)
        elif h.dim() == 2: # (B, D) or (B, T)
            return h.pow(2).mean(dim=1)
        else:
            return h.pow(2).mean()

    def _symbiotic_loss_with_peer_norm(self, pos_g: torch.Tensor, neg_g: torch.Tensor) -> torch.Tensor:
        """
        Symbiotic Loss + Peer Normalization
        Goodnessの平均を引くことで、特定のサンプルだけが勝ちすぎるのを防ぐ。
        """
        # 数値安定性のためのepsilon
        eps = 1e-6

        # Peer Normalization (Batch内の平均を引く)
        # これによりGoodnessが発散せず、相対的な「良さ」を学習する
        pos_g_mean = pos_g.mean().detach()
        neg_g_mean = neg_g.mean().detach()
        
        pos_g_norm = pos_g - pos_g_mean
        neg_g_norm = neg_g - neg_g_mean

        # Loss calculation (Softplus)
        # Positiveは閾値より大きく、Negativeは閾値より小さくしたい
        # log(1 + exp(...))
        loss_pos = torch.nn.functional.softplus(-(pos_g_norm - self.threshold))
        loss_neg = torch.nn.functional.softplus(neg_g_norm - self.threshold)
        
        return (loss_pos + loss_neg).mean()

    def _get_model_device(self) -> torch.device:
        try:
            if isinstance(self.model, nn.Module):
                return next(self.model.parameters()).device
            elif isinstance(self.model, AbstractNetwork):
                params = self.model.get_parameters()
                if params: return params[0].device
        except Exception:
            pass
        return torch.device("cpu")

    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        logger.info(f"🚀 Starting FF Epoch {self.current_epoch} (Temporal Contrast Mode)")
        total_loss = 0.0
        batch_count = 0
        device = self._get_model_device()
        
        for batch in data_loader:
            if isinstance(batch, dict):
                inputs = batch.get('input_ids', batch.get('input_images'))
            else:
                inputs, _ = batch
            
            if inputs is None: continue
            inputs = inputs.to(device)

            # Hard Negative 生成 (Temporal Shuffle適用)
            neg_inputs = self._generate_hard_negative(inputs)

            h_pos = inputs
            h_neg = neg_inputs
            
            batch_loss_sum = 0.0

            # --- Layer-wise Local Learning ---
            for i, layer in enumerate(self.trainable_layers):
                optimizer = self.layer_optimizers[i]
                
                # 1. Forward Passes
                h_pos_out = layer(h_pos)
                h_neg_out = layer(h_neg)

                # 出力の整形 (再帰的にスパイクやActivationsを探す)
                def _extract(out: Any) -> torch.Tensor:
                    if isinstance(out, tuple): 
                        return out[0]
                    if isinstance(out, dict): 
                        for key in ['spikes', 'activity', 'x']:
                            if key in out: return out[key]
                        return list(out.values())[0]
                    return out

                act_pos = _extract(h_pos_out)
                act_neg = _extract(h_neg_out)

                # 2. Goodness Calculation (Temporal)
                g_pos = self._calculate_temporal_goodness(act_pos)
                g_neg = self._calculate_temporal_goodness(act_neg)
                
                # 3. Loss with Peer Norm
                loss = self._symbiotic_loss_with_peer_norm(g_pos, g_neg)

                # 4. Local Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss_sum += loss.item()

                # 5. Detach for next layer
                # FFでは前の層の出力を次の層の入力として使う
                h_pos = act_pos.detach()
                h_neg = act_neg.detach()

            total_loss += batch_loss_sum
            batch_count += 1
        
        avg_loss = total_loss / max(1, batch_count)
        self.current_epoch += 1
        return {'loss': avg_loss}
