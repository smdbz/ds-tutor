import numpy as np
import pandas as pd
import plotly.express as px

from ..base import BaseTutor


class ScalingTutor(BaseTutor):
    def __init__(self):
        super().__init__(name="Feature Scaling Check")

    def check_condition(self, df: pd.DataFrame) -> bool:
        """Trigger if the max variance of any numeric column is 100x larger than the min variance."""
        numerics = df.select_dtypes(include=[np.number])
        if numerics.empty:
            return False

        variances = numerics.var()
        return (variances.max() / (variances.min() + 1e-5)) > 100

    def get_reasoning(self) -> str:
        return (
            "**High Variance Imbalance Detected.**\n"
            "Your numerical features are on drastically different scales. "
            "Algorithms that compute distance (like KNN or SVM) or use Gradient Descent "
            "(like Neural Networks or Logistic Regression) will heavily bias towards the features with larger numbers."
        )

    def get_math(self) -> str:
        # Using raw strings (r"") prevents Python from escaping the backslashes
        return r"""
        In Gradient Descent, the weight update rule is:
        $$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$
        If feature $x_j$ is on a massive scale, the gradient dominates the updates, 
        causing the algorithm to oscillate and struggle to converge.
        """

    def get_code_suggestion(self) -> str:
        return """
        ```python
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
    
        # Added StandardScaler to your pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ])```
        """

    def get_visual(self, df: pd.DataFrame):
        # A simple box plot to visually show the wildly different scales
        numerics = df.select_dtypes(include=[np.number])
        fig = px.box(numerics, title="Feature Scale Comparison (Notice the outliers/dominating features)")
        return fig
