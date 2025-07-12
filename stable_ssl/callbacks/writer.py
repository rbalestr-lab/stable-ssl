from pathlib import Path
from typing import Union

import torch
from lightning.pytorch import Callback, LightningModule
from loguru import logger as logging


class OnlineWriter(Callback):
    """Attaches an OnlineWriter callback."""

    def __init__(
        self,
        names: str,
        path: Union[str, Path],
        during: Union[str, list[str]],
        every_k_epochs: int = -1,
        save_last_epoch: bool = False,
        save_sanity_check: bool = False,
        all_gather: bool = True,
    ) -> None:
        super().__init__()
        logging.info("Setting up OnlineWriter callback")
        logging.info(f"\t- {names=}")
        logging.info(f"\t- {path=}")
        logging.info(f"\t- {during=}")
        logging.info(f"\t- {every_k_epochs=}")
        logging.info(f"\t- {save_last_epoch=}")
        logging.info(f"\t- {save_sanity_check=}")
        logging.info(f"\t- {all_gather=}")

        path = Path(path)
        if type(names) is str:
            names = [names]
        if type(during) is str:
            during = [during]
        self.names = names
        self.path = path
        self.during = during

        # -- writing conditions
        self.every_k_epochs = every_k_epochs
        self.save_last_epoch = save_last_epoch

        # -- writing in sanity check
        self.save_sanity_check = save_sanity_check
        self.is_sanity_check = False

        # -- writing device
        self.all_gather = all_gather

        if not path.is_dir():
            logging.warning(f"{path=} does not exist, creating it!")
            path.mkdir(parents=True, exist_ok=False)

    def on_sanity_check_start(self, trainer, pl_module):
        self.is_sanity_check = True

        if not self.save_sanity_check:
            logging.warning("OnlineWriter: skipping sanity check writing")

    def on_sanity_check_end(self, trainer, pl_module):
        self.is_sanity_check = False

    def is_writing_epoch(self, pl_module):
        current_epoch = pl_module.current_epoch
        max_epochs = pl_module.trainer.max_epochs

        # -- writing conditions
        save_every_epoch = self.every_k_epochs == -1
        save_k_epoch = (
            current_epoch % self.every_k_epochs == 0
            if self.every_k_epochs != 0
            else False
        )

        # last epoch condition
        is_last_epoch = current_epoch == max_epochs - 1
        save_last_epoch = self.save_last_epoch and is_last_epoch

        return any([save_every_epoch, save_k_epoch, save_last_epoch])

    def write_at_phase(
        self,
        pl_module,
        phase_name,
        outputs,
        batch_idx,
    ):
        # skip sanity check writing if necessary
        if self.is_sanity_check and not self.save_sanity_check:
            return

        # check if we are writing at this phase
        if not self.is_writing_epoch(pl_module) or phase_name not in self.during:
            return

        file_info = {
            "epoch": pl_module.current_epoch,
            "batch": batch_idx,
        }

        if not self.all_gather:
            file_info["device"] = pl_module.local_rank

        file_info = "_".join(f"{k}={v}" for k, v in file_info.items())
        filename = f"{phase_name}_{file_info}.pt"

        self.dump(pl_module, outputs, filename)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.write_at_phase(pl_module, "train", outputs, batch_idx)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.write_at_phase(pl_module, "predict", outputs, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.write_at_phase(pl_module, "test", outputs, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.write_at_phase(pl_module, "validation", outputs, batch_idx)

    def dump(self, pl_module: LightningModule, outputs: dict, filename: str):
        to_save = {}

        for name in self.names:
            if name not in outputs:
                msg = (
                    f"Asking to write {name} but not present "
                    f"in current batch {list(outputs.keys())}"
                )
                logging.error(msg)
                raise ValueError(msg)

            data = outputs[name]

            if self.all_gather:
                to_save[name] = pl_module.all_gather(data).cpu()
            else:
                to_save[name] = data.cpu()

        if not self.all_gather or pl_module.local_rank == 0:
            torch.save(to_save, self.path / filename)
