# ファイルパス: experiments/run_dummy_training.py
# (修正: mypyエラー解消)

import sys
import os
import logging
from typing import Tuple, Dict, Any, Optional, cast
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
except ImportError:
    Tensor = Any  # type: ignore[misc, assignment]
    Dataset = Any  # type: ignore[misc, assignment]
    DataLoader = Any  # type: ignore[misc, assignment]
    torch = Any  # type: ignore[assignment]

WANDB_AVAILABLE: bool = False
TENSORBOARD_AVAILABLE: bool = False
try:
    import wandb  # type: ignore[import-not-found]
    WANDB_AVAILABLE = True
except ImportError:
    pass
if not WANDB_AVAILABLE:
    try:
        # type: ignore[import-not-found]
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        pass

try:
    from snn_research.config.learning_config import PredictiveCodingConfig
    from snn_research.core.networks.sequential_snn_network import SequentialSNNNetwork
    from snn_research.training.base_trainer import AbstractTrainer, LoggerProtocol
    from snn_research.core.layers.lif_layer import LIFLayer
except ImportError as e:
    print(f"Error importing SNN modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger("P5_Script")


class ConsoleLogger(LoggerProtocol):
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        logger.info(f"[Epoch {step}] Metrics: {data}")


class TensorBoardLogger(LoggerProtocol):
    def __init__(self, log_dir: str = "workspace/runs/snn4_p5_dummy") -> None:
        self.writer = SummaryWriter(log_dir=log_dir)  # type: ignore

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        for k, v in data.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)  # type: ignore

    def close(self) -> None: self.writer.close()  # type: ignore


class DummySpikingDataset(Dataset):
    def __init__(self, num_samples: int, time_steps: int, features: int):
        self.num_samples = num_samples
        self.data: Any  # type: ignore
        self.targets: Any  # type: ignore

        if torch != Any:
            self.data = (torch.rand(
                num_samples, time_steps, features) > 0.8).float()
            self.targets = (torch.rand(num_samples) * 2).long()
        else:
            self.data = []
            self.targets = []

    def __len__(self) -> int: return self.num_samples
    def __getitem__(
        self, idx: int) -> Tuple[Tensor, Tensor]: return self.data[idx], self.targets[idx]


def main() -> None:
    if torch == Any:
        return

    config_dict: Dict[str, Any] = {
        "batch_size": 8, "time_steps": 20, "input_features": 10,
        "lif1_neurons": 32, "lif2_neurons": 2, "epochs": 3,
        "learning_rate": 0.01, "error_weight": 0.5,
        "lif_decay": 0.95, "lif_threshold": 1.0,
    }

    pc_config = PredictiveCodingConfig(
        learning_rate=config_dict["learning_rate"],
        error_weight=config_dict["error_weight"]
    )

    train_dataset = DummySpikingDataset(
        100, config_dict["time_steps"], config_dict["input_features"])
    train_loader = DataLoader(
        train_dataset, batch_size=config_dict["batch_size"], shuffle=True)
    eval_dataset = DummySpikingDataset(
        20, config_dict["time_steps"], config_dict["input_features"])
    eval_loader = DataLoader(
        eval_dataset, batch_size=config_dict["batch_size"], shuffle=False)

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
    if WANDB_AVAILABLE:
        wandb.init(project="SNN4_P5_Dummy", config=config_dict,
                   name="P5-2_run")  # type: ignore
    elif TENSORBOARD_AVAILABLE:
        logger_client = TensorBoardLogger()

    trainer = AbstractTrainer(model=model, logger_client=logger_client)

    logger.info(f"Starting training for {config_dict['epochs']} epochs...")

    for epoch in range(config_dict["epochs"]):
        model.reset_state()
        trainer.train_epoch(train_loader)
        model.reset_state()
        trainer.evaluate_epoch(eval_loader)

    if WANDB_AVAILABLE:
        wandb.finish()  # type: ignore
    elif TENSORBOARD_AVAILABLE:
        cast(TensorBoardLogger, logger_client).close()


if __name__ == "__main__":
    main()
