from .checkpoint_sklearn import SklearnCheckpoint
from .trainer_info import LoggingCallback, ModuleSummary, TrainerInfo


def default():
    return [
        # RichProgressBar(),
        LoggingCallback(),
        TrainerInfo(),
        SklearnCheckpoint(),
        ModuleSummary(),
    ]
