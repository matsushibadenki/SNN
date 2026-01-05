# ファイルパス: snn_research/validation/validator.py
# 日本語タイトル: SNN性能検証バリデータ
# 目的: モデルのベンチマーク結果(精度、エネルギー等)を目標値と比較し、Pass/Failを判定する。
# (修正: energy_ratioが0の場合のゼロ除算エラーを解消)

import logging
from typing import Dict, Any, Optional, List
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
        """
        # 修正: 型ヒントを明示して mypy エラー "object has no attribute append" を回避
        report: Dict[str, Any] = {
            "status": "PASS",
            "checks": [],
            "score": 0.0
        }

        checks: List[Dict[str, Any]] = []

        # 1. 精度の検証
        snn_acc = snn_metrics.get("accuracy", 0.0)
        ann_acc = ann_baseline_metrics.get(
            "accuracy", 1.0) if ann_baseline_metrics else 1.0

        acc_ratio = snn_acc / ann_acc if ann_acc > 0 else 0.0
        acc_pass = acc_ratio >= self.targets.accuracy.threshold_ratio

        checks.append({
            "name": "Accuracy Check",
            "snn_value": snn_acc,
            "target": f">= {self.targets.accuracy.threshold_ratio:.1%} of ANN ({ann_acc:.4f})",
            "passed": bool(acc_pass)
        })
        if not acc_pass:
            report["status"] = "FAIL"

        # 2. エネルギー効率の検証
        snn_energy = snn_metrics.get("estimated_energy_joules", float('inf'))
        ann_energy = ann_baseline_metrics.get(
            "estimated_energy_joules", 1.0) if ann_baseline_metrics else 1.0

        energy_ratio = snn_energy / \
            ann_energy if ann_energy > 0 else float('inf')
        energy_pass = energy_ratio <= self.targets.energy.max_ratio

        # 修正: energy_ratioが0の場合(スパイクなし等)のゼロ除算を防ぐ
        if energy_ratio > 1e-12:
            inverse_ratio_val = 1.0 / energy_ratio
            inverse_ratio_str = f"{inverse_ratio_val:.1f}"
        else:
            inverse_ratio_str = "Inf"

        checks.append({
            "name": "Energy Efficiency Check",
            "snn_value": f"{snn_energy:.2e} J",
            "target": f"<= {self.targets.energy.max_ratio:.1%} of ANN ({ann_energy:.2e} J)",
            "result_ratio": f"{energy_ratio:.4f} (1/{inverse_ratio_str}x)",
            "passed": bool(energy_pass)
        })
        if not energy_pass:
            report["status"] = "FAIL"

        # 3. スパイク率の検証
        spike_rate = snn_metrics.get("avg_spike_rate", 1.0)
        rate_pass = spike_rate <= self.targets.energy.target_spike_rate

        checks.append({
            "name": "Spike Rate Check",
            "snn_value": f"{spike_rate:.2%}",
            "target": f"<= {self.targets.energy.target_spike_rate:.1%}",
            "passed": bool(rate_pass)
        })

        report["checks"] = checks
        self.results = report
        return report

    def generate_markdown_summary(self) -> str:
        """検証結果のMarkdownサマリーを生成"""
        r = self.results
        if not r:
            return "No results available."

        status_icon = "✅" if r.get("status") == "PASS" else "❌"

        md = f"## {status_icon} SNN Performance Verification Report\n\n"
        md += f"**Overall Status:** {r.get('status', 'UNKNOWN')}\n\n"
        md += "| Metric | SNN Value | Target | Result |\n"
        md += "| :--- | :--- | :--- | :--- |\n"

        checks = r.get("checks", [])
        for c in checks:
            icon = "OK" if c.get("passed") else "NG"
            val = c.get("result_ratio", c.get("snn_value", "N/A"))
            if isinstance(val, float):
                val = f"{val:.4f}"
            md += f"| {c.get('name')} | {c.get('snn_value')} | {c.get('target')} | {icon} |\n"

        return md
