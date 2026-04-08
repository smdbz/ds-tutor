"""Backward-compatible imports for historical module paths."""

from ds_tutor.core import Experiment, Project
from ds_tutor.tutors import ExploratoryDataAnalysisTutor, MissingDataTutor, SkewnessTutor, TextDataTutor

__all__ = [
    "Project",
    "Experiment",
    "MissingDataTutor",
    "SkewnessTutor",
    "TextDataTutor",
    "ExploratoryDataAnalysisTutor",
]
