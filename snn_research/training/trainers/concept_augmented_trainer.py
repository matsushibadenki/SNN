# snn_research/training/trainers/concept_augmented_trainer.py
# 修正: Mypyの型不一致エラー(float + Tensor)を修正。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

from snn_research.core.snn_core import SNNCore
from snn_research.cognitive_architecture.neuro_symbolic_bridge import NeuroSymbolicBridge

logger = logging.getLogger(__name__)

class ConceptAugmentedTrainer:
    # ... (__init__等は変更なし) ...

    def __init__(
        self, 
        model: Union[SNNCore, nn.Module], 
        bridge: NeuroSymbolicBridge,
        learning_rate: float = 1e-3,
        concept_loss_weight: float = 1.0,
        temperature: float = 0.07,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.bridge = bridge
        self.device = device
        self.concept_loss_weight = concept_loss_weight
        self.temperature = temperature
        
        self.trainable_model: Any = model.model if isinstance(model, SNNCore) else model
        self.trainable_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.trainable_model.parameters(), lr=learning_rate)
        
        self.task_loss_fn = nn.CrossEntropyLoss()
        
        self.concept_prototypes: Dict[str, torch.Tensor] = {}
        self.momentum = 0.9

    def update_prototypes(self, concepts: List[str], features: torch.Tensor):
        with torch.no_grad():
            for i, concept in enumerate(concepts):
                feat = features[i].detach()
                if concept not in self.concept_prototypes:
                    self.concept_prototypes[concept] = feat
                else:
                    self.concept_prototypes[concept] = (
                        self.momentum * self.concept_prototypes[concept] + 
                        (1 - self.momentum) * feat
                    )
                self.concept_prototypes[concept] = F.normalize(
                    self.concept_prototypes[concept], p=2, dim=0
                )

    def prototype_loss(self, features: torch.Tensor, concepts: List[str]) -> torch.Tensor:
        # 初期値をTensorにする (Mypy対策)
        loss = torch.tensor(0.0, device=self.device)
        valid_count = 0
        
        features = F.normalize(features, p=2, dim=1)
        
        for i, concept in enumerate(concepts):
            if concept in self.concept_prototypes:
                proto = self.concept_prototypes[concept]
                sim = torch.dot(features[i], proto)
                loss = loss + (1.0 - sim) # In-place add (+=) may cause issues with gradients sometimes
                valid_count += 1
                
        if valid_count > 0:
            return loss / valid_count
        return loss # 0.0 tensor

    def train_step(self, specific_data: torch.Tensor, abstract_concepts: List[str], targets: torch.Tensor) -> Dict[str, float]:
        self.trainable_model.train()
        self.optimizer.zero_grad()
        
        specific_data = specific_data.to(self.device)
        targets = targets.to(self.device)
        batch_size = specific_data.size(0)

        # 1. Sensory
        if hasattr(self.trainable_model, 'forward_sensory'):
            sensory_output = self.trainable_model.forward_sensory(specific_data)
        else:
            sensory_output = self.trainable_model(specific_data)

        # 2. Conceptual
        concept_spikes = self.bridge.symbol_to_spike(abstract_concepts, batch_size=batch_size).to(self.device)
        if hasattr(self.trainable_model, 'forward_conceptual'):
            conceptual_output = self.trainable_model.forward_conceptual(concept_spikes)
        else:
            conceptual_output = concept_spikes

        # 3. Integration
        if hasattr(self.trainable_model, 'integrate'):
            integrated_output = self.trainable_model.integrate(sensory_output, conceptual_output)
        else:
            integrated_output = sensory_output

        loss_task = self.task_loss_fn(integrated_output, targets)

        # 4. Concept Loss
        # Mypy対策: 初期化をTensorに
        loss_concept: torch.Tensor = torch.tensor(0.0, device=self.device)
        
        if hasattr(self.trainable_model, 'get_internal_state'):
            visual_proj = self.trainable_model.get_internal_state()
            if hasattr(self.trainable_model, 'get_concept_projection'):
                concept_proj = self.trainable_model.get_concept_projection(conceptual_output)
            else:
                concept_proj = conceptual_output

            if visual_proj.shape == concept_proj.shape:
                visual_norm = F.normalize(visual_proj, p=2, dim=1)
                concept_norm = F.normalize(concept_proj, p=2, dim=1)
                logits = torch.matmul(visual_norm, concept_norm.T) / self.temperature
                labels = torch.arange(batch_size).to(self.device)
                
                # contrastive_loss メソッドをクラス内に追加するか、ここで計算
                # 簡易化のためここで計算
                loss_i2c = F.cross_entropy(logits, labels)
                loss_c2i = F.cross_entropy(logits.T, labels)
                loss_contrastive = (loss_i2c + loss_c2i) / 2
            else:
                loss_contrastive = torch.tensor(0.0, device=self.device)

            self.update_prototypes(abstract_concepts, concept_proj)
            loss_proto = self.prototype_loss(visual_proj, abstract_concepts)
            
            loss_concept = loss_contrastive + 0.5 * loss_proto

        total_loss = loss_task + (self.concept_loss_weight * loss_concept)

        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "task_loss": loss_task.item(),
            "concept_loss": loss_concept.item()
        }