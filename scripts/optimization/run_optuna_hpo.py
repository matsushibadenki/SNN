# ファイルパス: experiments/run_optuna_hpo.py
# (修正: mypyエラー解消)

import sys
import os
import logging
from typing import Tuple, Dict, Any, Optional
from collections import OrderedDict

project_root: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    Tensor = torch.Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any  # type: ignore[misc, assignment]
    Dataset = Any  # type: ignore[misc, assignment]
    DataLoader = Any  # type: ignore[misc, assignment]
    torch = Any  # type: ignore[assignment]

WANDB_AVAILABLE: bool = False
try:

    WANDB_AVAILABLE = True
except ImportError:
    pass

OPTUNA_AVAILABLE: bool = False
try:
    import optuna  # type: ignore[import-not-found]
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    Trial = Any  # type: ignore[misc, assignment]

try:
    from snn_research.config.learning_config import PredictiveCodingConfig
    from snn_research.core.networks.sequential_snn_network import SequentialSNNNetwork
    from snn_research.training.base_trainer import AbstractTrainer, LoggerProtocol
    from snn_research.core.layers.lif_layer import LIFLayer
except ImportError:
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("P5_HPO_Script")


class ConsoleLogger(LoggerProtocol):
    def log(self, data: Dict[str, Any],
            step: Optional[int] = None) -> None: pass


class DummySpikingDataset(Dataset):
    def __init__(self, num_samples: int, time_steps: int, features: int):
        self.num_samples = num_samples
        self.data: Any  # type: ignore
        self.targets: Any  # type: ignore

        if TORCH_AVAILABLE:
            self.data = (torch.rand(
                num_samples, time_steps, features) > 0.8).float()
            self.targets = (torch.rand(num_samples) * 2).long()
        else:
            self.data, self.targets = [], []

    def __len__(self) -> int: return self.num_samples
    def __getitem__(
        self, idx: int) -> Tuple[Tensor, Tensor]: return self.data[idx], self.targets[idx]


def objective(trial: Trial) -> float:
    global train_loader, eval_loader
    if train_loader is None or eval_loader is None:
        return -1.0

    config_dict: Dict[str, Any] = {
        "batch_size": 8, "time_steps": 20, "input_features": 10,
        "lif1_neurons": 32, "lif2_neurons": 2, "epochs": 3,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "error_weight": trial.suggest_float("error_weight", 0.1, 1.0),
        "lif_decay": trial.suggest_float("lif_decay", 0.8, 0.99),
        "lif_threshold": trial.suggest_float("lif_threshold", 0.8, 1.5),
    }

    pc_config = PredictiveCodingConfig(
        learning_rate=config_dict["learning_rate"],
        error_weight=config_dict["error_weight"]
    )

    lif1 = LIFLayer(
        input_features=config_dict["input_features"],
        neurons=config_dict["lif1_neurons"],
        learning_config=pc_config, name="lif1",
        decay=config_dict["lif_decay"], threshold=config_dict["lif_threshold"],
    )
    lif2 = LIFLayer(
        input_features=config_dict["lif1_neurons"],
        neurons=config_dict["lif2_neurons"],
        learning_config=pc_config, name="lif2",
        decay=config_dict["lif_decay"], threshold=config_dict["lif_threshold"],
    )
    lif1.build()
    lif2.build()

    # 修正: nn.Module として型付け
    layers_dict: OrderedDict[str, nn.Module] = OrderedDict(
        [("lif1", lif1), ("lif2", lif2)])
    model = SequentialSNNNetwork(layers=layers_dict)

    logger_client: LoggerProtocol = ConsoleLogger()
    trainer = AbstractTrainer(model=model, logger_client=logger_client)

    final_acc = 0.0
    for epoch in range(config_dict["epochs"]):
        model.reset_state()
        trainer.train_epoch(train_loader)
        model.reset_state()
        eval_metrics = trainer.evaluate_epoch(eval_loader)

        acc = eval_metrics.get("accuracy", 0.0)
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        final_acc = acc

    return final_acc


train_loader: Optional[DataLoader] = None
eval_loader: Optional[DataLoader] = None


def main_hpo() -> None:
    global train_loader, eval_loader
    if not TORCH_AVAILABLE or not OPTUNA_AVAILABLE:
        return

    train_dataset = DummySpikingDataset(100, 20, 10)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_dataset = DummySpikingDataset(20, 20, 10)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=5)

    logger.info(f"Best accuracy: {study.best_value}")


if __name__ == "__main__":
    main_hpo()
