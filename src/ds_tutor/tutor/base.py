from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict


class BaseTutor(ABC):
    """
    The blueprint for all the ds_tutor tutors.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def check_condition(self, df: pd.DataFrame, pipeline=None) -> bool:
        """Evaluates the data/model to see if this advice is needed."""
        pass

    @abstractmethod
    def get_math(self) -> str:
        """Returns the LaTeX mathematical justification."""
        pass

    @abstractmethod
    def get_reasoning(self) -> str:
        """Returns the plain English explanation."""
        pass

    @abstractmethod
    def get_visual(self, df: pd.DataFrame):
        """Returns a Plotly/Matplotlib figure object to display."""
        pass

    @abstractmethod
    def get_code_suggestion(self) -> str:
        """Returns the generated sklearn Python code to copy-paste."""
        pass
