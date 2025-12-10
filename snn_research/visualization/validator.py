# ファイルパス: snn_research/validation/validator.py
# タイトル: SNN性能検証バリデータ
# 機能説明: 
#   モデルの実行結果を目標値(targets)と比較し、Pass/Failを判定する。
#   エネルギー効率、精度、スパイク率を包括的に評価する。

import torch
import logging
from typing import Dict, Any, Optional
from snn_research.metrics.energy import EnergyMetrics
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class PerformanceValidator:
    """
    SNNモデルの性能を定義された目標値に対して検証するクラス。
    """
    def __init__(self, target_config: DictConfig):
        self.targets = target_config.targets
        self.results: Dict[str, Any] = {}

    def validate(
        self, 
        snn_metrics: Dict[str, float], 
        ann_baseline_metrics: Optional[Dict[str, float]] = None,
        model_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        検証を実行し、詳細なレポートデータを返す。
        
        Args:
            snn_metrics: SNNの評価結果 (accuracy, total_spikesなど)
            ann_baseline_metrics: 比較対象のANN評価結果 (accuracy, energyなど)
            model_stats: モデルの統計情報 (パラメータ数、ニューロン数など)
        """
        report = {
            "status": "PASS",
            "checks": [],
            "score": 0.0
        }
        
        # 1. 精度の検証
        snn_acc = snn_metrics.get("accuracy", 0.0)
        ann_acc = ann_baseline_metrics.get("accuracy", 1.0) if ann_baseline_metrics else 1.0
        
        acc_ratio = snn_acc / ann_acc if ann_acc > 0 else 0.0
        acc_pass = acc_ratio >= self.targets.accuracy.threshold_ratio
        
        report["checks"].append({
            "name": "Accuracy Check",
            "snn_value": snn_acc,
            "target": f">= {self.targets.accuracy.threshold_ratio:.1%} of ANN ({ann_acc:.4f})",
            "passed": bool(acc_pass)
        })
        if not acc_pass: report["status"] = "FAIL"

        # 2. エネルギー効率の検証
        # EnergyMetricsを使用して計算済みと仮定、またはここで再計算
        snn_energy = snn_metrics.get("estimated_energy_joules", float('inf'))
        ann_energy = ann_baseline_metrics.get("estimated_energy_joules", 1.0) if ann_baseline_metrics else 1.0
        
        energy_ratio = snn_energy / ann_energy if ann_energy > 0 else float('inf')
        energy_pass = energy_ratio <= self.targets.energy.max_ratio
        
        report["checks"].append({
            "name": "Energy Efficiency Check",
            "snn_value": f"{snn_energy:.2e} J",
            "target": f"<= {self.targets.energy.max_ratio:.1%} of ANN ({ann_energy:.2e} J)",
            "result_ratio": f"{energy_ratio:.4f} (1/{1/energy_ratio:.1f}x)",
            "passed": bool(energy_pass)
        })
        if not energy_pass: report["status"] = "FAIL" # 警告にとどめる場合は条件分岐

        # 3. スパイク率の検証
        spike_rate = snn_metrics.get("avg_spike_rate", 1.0)
        rate_pass = spike_rate <= self.targets.energy.target_spike_rate
        
        report["checks"].append({
            "name": "Spike Rate Check",
            "snn_value": f"{spike_rate:.2%}",
            "target": f"<= {self.targets.energy.target_spike_rate:.1%}",
            "passed": bool(rate_pass)
        })

        self.results = report
        return report

    def generate_markdown_summary(self) -> str:
        """検証結果のMarkdownサマリーを生成"""
        r = self.results
        status_icon = "✅" if r["status"] == "PASS" else "❌"
        
        md = f"## {status_icon} SNN Performance Verification Report\n\n"
        md += f"**Overall Status:** {r['status']}\n\n"
        md += "| Metric | SNN Value | Target | Result |\n"
        md += "| :--- | :--- | :--- | :--- |\n"
        
        for c in r["checks"]:
            icon = "OK" if c["passed"] else "NG"
            val = c.get("result_ratio", c["snn_value"])
            if isinstance(val, float): val = f"{val:.4f}"
            md += f"| {c['name']} | {c['snn_value']} | {c['target']} | {icon} |\n"
            
        return md