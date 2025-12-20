# ファイルパス: snn_research/training/trainers/forward_forward.py
# 日本語タイトル: Forward-Forward アルゴリズムトレーナー (Ver 3.0 "Deep Insight")
# 機能説明:
#   1. Hard Negative Mixing: 文脈を破壊する高度な偽データ生成。
#   2. Temporal Energy: スパイクの時系列エネルギーに基づくGoodness計算。
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
    Forward-Forward Algorithm (v3.0 Deep Insight)
    
    Quality Enhancements:
    - Hard Negative Mixing: より難易度の高い「偽データ」を生成し、特徴抽出能力を高める。
    - Peer Normalization: Goodnessの暴走を防ぎ、学習を安定化させる。
    - Temporal Awareness: SNNの時間方向の情報を統合して評価する。
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
        """レイヤー探索とオプティマイザ設定（Ver 2.1準拠＋強化）"""
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
                optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=1e-4)
                self.layer_optimizers.append(optimizer)
        
        logger.info(f"🔍 FF Trainer: Configured {len(self.trainable_layers)} layers for Deep Insight training.")

    def _generate_hard_negative(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Hard Negative Mining:
        単なるシャッフルではなく、バッチ内の他のサンプルと「混ぜる」ことで、
        より見分けにくい（学習効果の高い）偽データを生成する。
        """
        batch_size = inputs.size(0)
        neg_inputs = inputs.clone()
        device = inputs.device

        # Case A: テキスト/シーケンス (Batch, SeqLen)
        if inputs.dtype == torch.long or (inputs.dim() == 3 and inputs.size(1) > 1):
            # Token Mixing: 別のサンプルのトークンを30%ほど混入させる
            perm_indices = torch.randperm(batch_size, device=device)
            source_samples = inputs[perm_indices]
            
            # マスク生成 (30%を置き換え)
            mask_prob = 0.3
            mask = torch.rand(inputs.shape, device=device) < mask_prob
            
            neg_inputs[mask] = source_samples[mask]
            return neg_inputs

        # Case B: 画像 (Batch, C, H, W)
        elif inputs.dim() == 4:
            # Patch Shuffle: 画像を4分割してシャッフルするなど（ここでは簡易版としてチャネルシャッフル）
            # またはバッチシャッフル
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

        # 時間次元がある場合
        if h.dim() == 3: 
            # 二乗してから時間平均 -> 次元の平均
            # これにより「強く発火している時間帯」を評価する
            return h.pow(2).mean(dim=1).mean(dim=1)
        elif h.dim() == 2:
            return h.pow(2).mean(dim=1)
        else:
            return h.pow(2).mean()

    def _symbiotic_loss_with_peer_norm(self, pos_g: torch.Tensor, neg_g: torch.Tensor) -> torch.Tensor:
        """
        Symbiotic Loss + Peer Normalization
        Goodnessの平均を引くことで、特定のサンプルだけが勝ちすぎるのを防ぐ。
        """
        # Peer Normalization (Batch内の平均を引く)
        # これによりGoodnessが発散せず、相対的な「良さ」を学習する
        pos_g_norm = pos_g - pos_g.mean().detach()
        neg_g_norm = neg_g - neg_g.mean().detach()

        # Loss calculation (Softplus)
        # Positiveは閾値より大きく、Negativeは閾値より小さくしたい
        loss_pos = torch.log1p(torch.exp(-(pos_g_norm - self.threshold)))
        loss_neg = torch.log1p(torch.exp(neg_g_norm - self.threshold))
        
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
        logger.info(f"🚀 Starting FF Epoch {self.current_epoch} (Deep Insight Mode)")
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

            # Hard Negative 生成
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
                        # SNNでは (logits, spikes, mem) の順が多い。Spikesを優先したい場合もあるが、
                        # FFでは「次の層への入力」をGoodnessとするのが基本。
                        # 通常は out[0] が出力テンソル。
                        return out[0]
                    if isinstance(out, dict): 
                        # 'spikes' があればそれを優先、なければ 'activity', 'x'
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
                h_pos = act_pos.detach()
                h_neg = act_neg.detach()

            total_loss += batch_loss_sum
            batch_count += 1
        
        avg_loss = total_loss / max(1, batch_count)
        self.current_epoch += 1
        return {'loss': avg_loss}