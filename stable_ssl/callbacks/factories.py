from .trainer_info import LoggingCallback, TrainerInfo, ModuleSummary
from .checkpoint_sklearn import SklearnCheckpoint
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks import RichModelSummary


def default():
    return [
        # RichProgressBar(),
        LoggingCallback(),
        TrainerInfo(),
        SklearnCheckpoint(),
        ModuleSummary(),
    ]
