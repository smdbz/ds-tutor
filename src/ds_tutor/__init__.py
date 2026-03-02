from .core import DsTutor
from .tutor.imputer_tutor import ImputerTutor
from .tutor.scaling_tutor import ScalingTutor

__all__ = ["DsTutor", "ImputerTutor", "ScalingTutor", "hello"]

def hello() -> str:
    return "Hello from ds-tutor!"
