# ファイルパス: snn_research/training/trainers/multimodal_trainer.py
# 日本語タイトル: Multimodal Trainer (VLM Optimization)
# 目的・内容:
#   SpikingVLMモデル専用の学習トレーナー。
#   視覚と言語のアライメント（Contrastive Loss）と、
#   テキスト生成（Language Modeling Loss）のマルチタスク学習を制御する。
#   勾配クリッピングや学習率スケジューリングも管理する。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm

from snn_research.models.transformer.spiking_vlm import SpikingVLM
# 既存のBaseTrainerがあれば継承、なければ独立クラスとして実装
# ここでは独立性が高い実装とするが、インターフェースは一般的なTrainerに合わせる

logger = logging.getLogger(__name__)

class MultimodalVLMTrainer:
    """
    SpikingVLM (Vision-Language Model) 用の学習トレーナー。
    Contrastive Loss (CLIP) と Generative Loss (Captioning) を統合して学習する。
    """

    def __init__(
        self,
        model: SpikingVLM,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Dict[str, Any] = {}
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Loss Weights
        self.lambda_align = config.get("lambda_align", 1.0) # Contrastive Loss weight
        self.lambda_gen = config.get("lambda_gen", 1.0)     # Generation Loss weight
        
        # Generation Loss (Cross Entropy)
        # ignore_indexはパディングトークンID（通常0または-100）を指定
        self.criterion_gen = nn.CrossEntropyLoss(ignore_index=0)
        
        self.grad_clip = config.get("grad_clip", 1.0)
        self.accumulation_steps = config.get("accumulation_steps", 1)
        
        logger.info(f"🚀 MultimodalVLMTrainer initialized on {device}.")
        logger.info(f"   Weights -> Align: {self.lambda_align}, Gen: {self.lambda_gen}")

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        1エポック分の学習を実行する。
        """
        self.model.train()
        total_loss = 0.0
        total_align_loss = 0.0
        total_gen_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Batch Data Handling
            # 想定: batch = {'images': ..., 'input_ids': ..., 'labels': ...}
            images = batch.get('images').to(self.device)
            input_ids = batch.get('input_ids').to(self.device)
            
            # Labels for generation (shifted input_ids or explicit labels)
            labels = batch.get('labels', None)
            if labels is None:
                # 自己回帰学習のため、input_idsを1つずらしてラベルとする簡易実装
                # [B, Seq] -> Labels
                labels = input_ids.clone()
                # 実際にはPADトークンの処理などが必要だが、ここでは簡易化
            labels = labels.to(self.device)
            
            # --- Forward Pass ---
            # SpikingVLM.forward() -> returns dict with logits, alignment_loss, etc.
            outputs = self.model(images, input_ids)
            
            # --- Loss Calculation ---
            # 1. Alignment Loss (CLIP style)
            align_loss = outputs["alignment_loss"]
            
            # 2. Generation Loss (Next Token Prediction)
            logits = outputs["logits"] # [B, T, Vocab]
            
            # Dimension adjustment for CrossEntropyLoss
            # Logits: [B, T, C] -> [B*T, C], Labels: [B, T] -> [B*T]
            # ここでは系列長を合わせる必要がある
            # SNNの出力系列長とテキスト入力系列長が一致している前提
            if logits.shape[1] != labels.shape[1]:
                 # 簡易的なリサイズ（本来はAttention Mask等で厳密にやるべき）
                 min_len = min(logits.shape[1], labels.shape[1])
                 logits = logits[:, :min_len, :]
                 labels = labels[:, :min_len]
            
            gen_loss = self.criterion_gen(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            # Total Loss
            loss = (self.lambda_align * align_loss) + (self.lambda_gen * gen_loss)
            loss = loss / self.accumulation_steps
            
            # --- Backward Pass ---
            loss.backward()
            
            if (step + 1) % self.accumulation_steps == 0:
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Logging
            current_loss = loss.item() * self.accumulation_steps
            total_loss += current_loss
            total_align_loss += align_loss.item()
            total_gen_loss += gen_loss.item()
            
            progress_bar.set_postfix({
                "Loss": f"{current_loss:.4f}",
                "Align": f"{align_loss.item():.4f}",
                "Gen": f"{gen_loss.item():.4f}"
            })
            
        avg_loss = total_loss / len(dataloader)
        avg_align = total_align_loss / len(dataloader)
        avg_gen = total_gen_loss / len(dataloader)
        
        logger.info(f"Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f} (Align: {avg_align:.4f}, Gen: {avg_gen:.4f})")
        
        return {
            "train_loss": avg_loss,
            "train_align_loss": avg_align,
            "train_gen_loss": avg_gen
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        検証ループ。
        """
        self.model.eval()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False):
            images = batch.get('images').to(self.device)
            input_ids = batch.get('input_ids').to(self.device)
            labels = batch.get('labels', input_ids.clone()).to(self.device)
            
            outputs = self.model(images, input_ids)
            
            align_loss = outputs["alignment_loss"]
            logits = outputs["logits"]
            
            if logits.shape[1] != labels.shape[1]:
                 min_len = min(logits.shape[1], labels.shape[1])
                 logits = logits[:, :min_len, :]
                 labels = labels[:, :min_len]

            gen_loss = self.criterion_gen(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            loss = (self.lambda_align * align_loss) + (self.lambda_gen * gen_loss)
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Validation Epoch {epoch}. Avg Loss: {avg_loss:.4f}")
        
        return {"val_loss": avg_loss}

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"💾 Checkpoint saved to {path}")