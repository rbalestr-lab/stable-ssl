from .checkpoint_sklearn import SklearnCheckpoint
from .teacher_student import TeacherStudentCallback
from .trainer_info import LoggingCallback, ModuleSummary, TrainerInfo


def default(pl_module):
    callbacks = [
        # RichProgressBar(),
        LoggingCallback(),
        TrainerInfo(),
        SklearnCheckpoint(),
        ModuleSummary(),
    ]

    # Auto-detect TeacherStudentWrapper and add callback if needed
    for module in pl_module.modules():
        if hasattr(module, "update_teacher") and hasattr(module, "teacher"):
            callbacks.append(TeacherStudentCallback())
            break

    return callbacks
