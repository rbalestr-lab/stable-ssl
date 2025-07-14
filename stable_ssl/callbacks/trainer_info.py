import lightning.pytorch as pl
import torch
from lightning.pytorch import Callback
from loguru import logger as logging
from prettytable import PrettyTable
from pytorch_lightning.utilities import rank_zero_only

from ..data.module import DataModule


class ModuleSummary(pl.Callback):
    """Callback for logging module summaries in a formatted table."""

    @rank_zero_only
    def setup(self, trainer, pl_module, stage):
        headers = [
            "Module",
            "Trainable parameters",
            "Non Trainable parameters",
            "Uninitialized parameters",
            "Buffers",
        ]
        table = PrettyTable()
        table.field_names = headers
        table.align["Module"] = "l"
        table.align["Trainable parameters"] = "r"
        table.align["Non Trainable parameters"] = "r"
        table.align["Uninitialized parameters"] = "r"
        table.align["Buffers"] = "r"
        logging.info("PyTorch Modules:")
        for name, module in pl_module.named_modules():
            num_trainable = 0
            num_nontrainable = 0
            num_buffer = 0
            num_uninitialized = 0
            for p in module.parameters():
                if isinstance(p, torch.nn.parameter.UninitializedParameter):
                    n = 0
                    num_uninitialized += 1
                else:
                    n = p.numel()
                if p.requires_grad:
                    num_trainable += n
                else:
                    num_nontrainable += n
            for p in module.buffers():
                if isinstance(p, torch.nn.parameter.UninitializedBuffer):
                    n = 0
                    num_uninitialized += 1
                else:
                    n = p.numel()
                num_buffer += n
            table.add_row(
                [name, num_trainable, num_nontrainable, num_uninitialized, num_buffer]
            )
        print(table)

        return super().setup(trainer, pl_module, stage)


class LoggingCallback(pl.Callback):
    """Callback for logging validation metrics in a formatted table."""

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                table.add_row(
                    [
                        "\033[0;34;40m" + key + "\033[0m",
                        "\033[0;32;40m" + str(metrics[key].item()) + "\033[0m",
                    ]
                )
        print(table)


class TrainerInfo(Callback):
    """Callback for linking trainer to DataModule and providing extra information."""

    def setup(self, trainer, pl_module, stage):
        logging.info("\t linking trainer to DataModule! 🔧")
        if not isinstance(trainer.datamodule, DataModule):
            logging.warning("Using a custom DataModule, won't have extra info!")
            return
        trainer.datamodule.set_pl_trainer(trainer)
        return super().setup(trainer, pl_module, stage)
