# snn_research/cli/recipe_commands.py

import click
from snn_research.recipes.mnist import run_mnist_training
from snn_research.recipes.cifar10 import run_cifar10_training

@click.group(name="run")
def recipe_cli():
    """定義済みレシピ(特定の学習スクリプト)の実行"""
    pass

@recipe_cli.command(name="mnist")
@click.option('--epochs', default=10, help="学習エポック数")
@click.option('--batch-size', default=128, help="バッチサイズ")
def cmd_mnist(epochs, batch_size):
    """MNIST SNN学習レシピを実行"""
    run_mnist_training(config_override={"epochs": epochs, "batch_size": batch_size})

@recipe_cli.command(name="cifar10")
@click.option('--epochs', default=50, help="学習エポック数")
@click.option('--batch-size', default=64, help="バッチサイズ")
def cmd_cifar10(epochs, batch_size):
    """CIFAR-10 SEW-ResNet学習レシピを実行"""
    run_cifar10_training(epochs=epochs, batch_size=batch_size)