# snn_research/cli/demo_commands.py

import click
import logging
from snn_research.scenarios.brain_v14 import run_scenario as run_v14

# 必要に応じて他のデモもインポート可能
# from snn_research.scenarios.brain_v16 import run_scenario as run_v16

logger = logging.getLogger("DemoCLI")

@click.group(name="demo")
def demo_cli():
    """各種デモ・シナリオの実行"""
    pass

@demo_cli.command(name="brain-v14")
@click.option('--config', default="configs/experiments/brain_v14_config.yaml")
def cmd_brain_v14(config):
    """Brain v14.0 シミュレーションを実行"""
    run_v14(config_path=config)

@demo_cli.command(name="brain-v16")
def cmd_brain_v16():
    """Brain v16.3 デモを実行 (Placeholder)"""
    logger.info("🚀 Running Brain v16 Demo (Not fully migrated yet)")
    # 将来的に v16 のロジックもここに移行する
    pass