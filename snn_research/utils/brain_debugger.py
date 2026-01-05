# ファイルパス: snn_research/utils/brain_debugger.py
# Title: Brain Debugger & Concept Visualizer
# Description:
#   ニューロンの発火状況を監視し、現在活性化している「概念」を特定・表示するツール。
#   スパイク活動 -> 概念マップ への変換を行う。

import torch
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("BrainDebugger")

class BrainDebugger:
    """
    SNNの内部状態を可視化・言語化するデバッガ。
    """
    def __init__(self, vocabulary: Optional[Dict[int, str]] = None):
        self.active_concepts: Dict[str, float] = {}
        self.history: List[str] = []
        self.concept_map = {
            0: "Self", 1: "User", 2: "Food", 3: "Danger",
            4: "Curiosity", 5: "Sleep", 6: "Logic", 7: "Emotion"
        }
        # ... (以下変更なし)

    def analyze_activity(self, spike_tensor: torch.Tensor, region_name: str) -> str:
        if spike_tensor.numel() == 0:
            return f"[{region_name}] No activity."

        firing_rate = spike_tensor.float().mean().item()
        
        if spike_tensor.dim() > 1:
            activity_profile = spike_tensor.float().mean(dim=[0, 1])
        else:
            activity_profile = spike_tensor.float()

        top_indices = torch.topk(activity_profile, k=min(3, len(activity_profile))).indices.tolist()
        
        active_concepts = []
        for idx in top_indices:
            concept = self.concept_map.get(idx % 8, f"Feature_{idx}")
            intensity = activity_profile[idx].item()
            if intensity > 0.01:
                active_concepts.append(f"{concept}({intensity:.2f})")

        status_msg = f"[{region_name}] Activity: {firing_rate:.3f} Hz | Concepts: {', '.join(active_concepts)}"
        
        if firing_rate > 0.05:
            logger.info(f"🧠 DEBUG: {status_msg}")
            
        return status_msg

    def explain_thought_process(self, input_text: str, output_text: str, astrocyte_status: Dict) -> str:
        energy = astrocyte_status.get('metrics', {}).get('current_energy', 0)
        fatigue = astrocyte_status.get('metrics', {}).get('fatigue_index', 0)
        
        explanation = "\n🔍 --- Thought Process Analysis ---\n"
        explanation += f"   Input Stimulus: '{input_text}'\n"
        
        bias = "Neutral"
        if energy < 200:
            bias = "Lazy (Low Energy)"
        elif fatigue > 80:
            bias = "Incoherent (Fatigued)"
        elif "Apple" in input_text or "Food" in input_text:
            bias = "Reward Seeking"
            
        explanation += f"   Internal State: {bias} (Energy: {energy:.0f})\n"
        explanation += f"   Generated Response: '{output_text}'\n"
        explanation += "   ---------------------------------"
        
        print(explanation)
        return explanation