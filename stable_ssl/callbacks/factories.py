from .checkpoint_sklearn import SklearnCheckpoint, WandbCheckpoint
from .teacher_student import TeacherStudentCallback
from .trainer_info import LoggingCallback, ModuleSummary, TrainerInfo, SLURMInfo


def default(pl_module=None):
    callbacks = [
        # RichProgressBar(),
        LoggingCallback(),
        TrainerInfo(),
        SklearnCheckpoint(),
        WandbCheckpoint(),
        ModuleSummary(),
        SLURMInfo(),
    ]

    # Auto-detect TeacherStudentWrapper and add callback if needed
    if pl_module is not None:
        for module in pl_module.modules():
            if hasattr(module, "update_teacher") and hasattr(module, "teacher"):
                callbacks.append(TeacherStudentCallback())
                break

    return callbacks
