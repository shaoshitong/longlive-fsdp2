from .distillation import Trainer as ScoreDistillationTrainer
from .distillation_ovi import Trainer as ScoreDistillationOviTrainer
__all__ = [
    "ScoreDistillationTrainer",
    "ScoreDistillationOviTrainer"
]
