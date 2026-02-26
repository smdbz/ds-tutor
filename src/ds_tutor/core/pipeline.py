from sklearn.pipeline import Pipeline
from IPython.display import display, Markdown
import pandas as pd

from ds_tutor.tutor.imputer_tutor import ImputerTutor
from ds_tutor.tutor.scaling_tutor import ScalingTutor

class TutorPipeline(Pipeline):
    """
    A wrapper around sklearn.pipeline.Pipeline that automatically
    evaluates the dataset and provides interactive tutoring/advice
    in Jupyter Notebooks before fitting the pipeline.
    """
    
    def __init__(self, steps, *, memory=None, verbose=False, tutors=None):
        super().__init__(steps, memory=memory, verbose=verbose)
        # Initialize the available tutors
        self.tutors = tutors if tutors is not None else [
            ImputerTutor(),
            ScalingTutor()
        ]
        
    def fit(self, X, y=None, **fit_params):
        """
        Fits the pipeline, but first evaluates X with the tutors
        and displays advice if any issues or edge cases are found.
        """
        # We only evaluate if X is a pandas DataFrame, as our tutors expect it
        if isinstance(X, pd.DataFrame):
            self._run_tutors(X)
            
        return super().fit(X, y, **fit_params)
        
    def _run_tutors(self, X: pd.DataFrame):
        for tutor in self.tutors:
            if tutor.check_condition(X, pipeline=self):
                # Display the advice using IPython.display
                display(Markdown(f"### 🧑‍🏫 ds-tutor: {tutor.name}"))
                display(Markdown(tutor.get_reasoning()))
                
                math_content = tutor.get_math()
                if math_content:
                    display(Markdown(math_content))
                
                code_content = tutor.get_code_suggestion()
                if code_content:
                    display(Markdown(code_content))
                
                visual = tutor.get_visual(X)
                if visual is not None:
                    # Plotly figures have a .show() method which renders nicely in Jupyter
                    try:
                        visual.show()
                    except Exception as e:
                        # Fallback just in case
                        display(Markdown(f"*Could not display visual: {e}*"))
                
                display(Markdown("---"))
