from typing import List, Optional

import pandas as pd
from IPython.display import display, Markdown

from .base import BaseTutor
from .tutor.imputer_tutor import ImputerTutor
from .tutor.scaling_tutor import ScalingTutor


class DsTutor:
    """
    A standalone class that automatically evaluates a dataset 
    and provides interactive tutoring/advice in Jupyter Notebooks.
    """

    def __init__(self, tutors: Optional[List[BaseTutor]] = None):
        self.tutors = tutors if tutors is not None else [
            ImputerTutor(),
            ScalingTutor()
        ]

    def evaluate(self, X: pd.DataFrame):
        """
        Evaluates X with the tutors and displays advice if any issues 
        or edge cases are found.
        """
        if not isinstance(X, pd.DataFrame):
            return

        for tutor in self.tutors:
            if tutor.check_condition(X):
                # Display the advice using IPython.display
                display(Markdown(f"### \U0001F9D1\u200D\U0001F3EB ds-tutor: {tutor.name}"))
                display(Markdown(tutor.get_reasoning()))

                math_content = tutor.get_math()
                if math_content:
                    display(Markdown(math_content))

                code_content = tutor.get_code_suggestion()
                if code_content:
                    display(Markdown(code_content))

                visual = tutor.get_visual(X)
                if visual is not None:
                    try:
                        visual.show()
                    except Exception as e:
                        display(Markdown(f"*Could not display visual: {e}*"))

                display(Markdown("---"))
