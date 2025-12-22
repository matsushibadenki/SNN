# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel (属性追加版)

import logging
import torch
from typing import Dict, Any
from torchvision import transforms

logger = logging.getLogger(__name__)

class ArtificialBrain:
    def __init__(self, **kwargs: Any):
        # 既存の初期化...
        self.visual = kwargs.get('visual_cortex')
        self.state = "AWAKE"
        self.cycle_count = 0
        
        # [追加] 空間認識デモ等で使用される画像変換プロセッサ
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        logger.info("ArtificialBrain Kernel v20.1 initialized.")

    # 既存の run_cognitive_cycle 等のメソッド...
    def calculate_uncertainty(self, result: Any) -> float:
        """
        [精度向上] エントロピーと最大確率のギャップに基づくメタ認知。
        System 2 (熟慮) を起動すべきか判断する指標。
        """
        if not isinstance(result, torch.Tensor):
            return 0.5
            
        with torch.no_grad():
            # 数値的安定性の確保
            logits = result.float()
            probs = torch.softmax(logits, dim=-1)
            
            # 1. シャノンエントロピー
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            normalized_entropy = min(1.0, float(entropy / 2.3)) # ln(10)で正規化
            
            # 2. Confidence Gap (最大確率の低さ)
            max_prob = torch.max(probs).item()
            confidence_score = 1.0 - max_prob
            
            # 統合スコア (高いほど不確実)
            uncertainty = (normalized_entropy + confidence_score) / 2.0
            return float(uncertainty)

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        認知サイクルの実行 (知覚 -> 監視 -> 制御)。
        """
        self.cycle_count += 1
        perception_result = None
        uncertainty = 0.0

        # 1. 知覚フェーズ
        if self.visual is not None and hasattr(self.visual, 'forward'):
            try:
                perception_result = self.visual(raw_input)
                uncertainty = self.calculate_uncertainty(perception_result)
            except Exception as e:
                logger.error(f"Perception failed at cycle {self.cycle_count}: {e}")

        # 2. アストロサイトによる代謝・疲労蓄積 
        if self.astrocyte and hasattr(self.astrocyte, 'accumulate_fatigue'):
            # 不確実性が高い（深く考えている）ほど疲労が溜まりやすい
            energy_demand = 1.0 + (uncertainty * 2.0) 
            self.astrocyte.accumulate_fatigue(0.2 * energy_demand)

        # 3. 倫理ガードレール (リアルタイム監査)
        if self.guardrail and hasattr(self.guardrail, 'check_safety'):
            try:
                self.guardrail.check_safety(perception_result)
            except Exception as e:
                logger.warning(f"Safety intercept: {e}")
                return {"cycle": self.cycle_count, "status": "BLOCKED", "reason": str(e)}

        # ステータスの同期
        current_status = self.get_status()

        return {
            "cycle": self.cycle_count,
            "status": "SUCCESS",
            "uncertainty": uncertainty,
            "state": self.state,
            "astrocyte": current_status["astrocyte"],
            "result": "Cognitive cycle completed."
        }

    def sleep_cycle(self) -> None:
        """
        🌙 睡眠サイクル。
        記憶の固定化（Consolidation）とアストロサイトのリセット。
        """
        if self.state == "SLEEPING":
            return

        logger.info(f"🛌 Cycle {self.cycle_count}: Entering Sleep state...")
        self.state = "SLEEPING"
        
        try:
            # 記憶固定化 (System 2 の思考を SNN へ転送)
            if self.sleep_manager and hasattr(self.sleep_manager, 'consolidate_memory'):
                self.sleep_manager.consolidate_memory()
                
            # エネルギー代謝の恒常性回復
            if self.astrocyte and hasattr(self.astrocyte, 'replenish_energy'):
                self.astrocyte.replenish_energy(1000.0)
                
        finally:
            self.state = "AWAKE"
            logger.info("☀️ Brain restored to AWAKE state.")

    def get_brain_status(self) -> Dict[str, Any]:
        """run_brain_v16_demo.py 等のエイリアス。"""
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """
        統合診断レポート。
        [修正] デモスクリプトの KeyError: 'status' を防ぐための厳密な構造。
        """
        energy = getattr(self.astrocyte, 'energy', 1000.0) if self.astrocyte else 1000.0
        fatigue = getattr(self.astrocyte, 'fatigue_toxin', 0.0) if self.astrocyte else 0.0
        
        # 疲労度に基づくステータス判定
        astro_status = "NORMAL"
        if fatigue > 50.0: astro_status = "WARNING"
        if fatigue > 80.0: astro_status = "CRITICAL"

        return {
            "state": self.state,
            "cycle": self.cycle_count,
            "astrocyte": {
                "status": astro_status,
                "energy_percent": (energy / 1000.0) * 100.0,
                "fatigue": fatigue,
                "metrics": {"energy_level": energy}
            }
        }
