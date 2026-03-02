import pandas as pd
import plotly.express as px

from ..base import BaseTutor


class ImputerTutor(BaseTutor):
    def __init__(self):
        super().__init__(name="Missing Value Check")

    def check_condition(self, df: pd.DataFrame) -> bool:
        """Trigger if the dataframe contains any null values."""
        if df.empty:
            return False
        return df.isnull().values.any()

    def get_reasoning(self) -> str:
        return (
            "**Missing Data Detected.**\n"
            "Your dataset contains missing values (NaNs). "
            "Most scikit-learn algorithms (e.g., SVM, Random Forest, Logistic Regression) "
            "cannot handle missing values and will throw an error during training.\n"
            "Dropping rows reduces your training data size, which can harm performance. "
            "Imputation (filling in the missing values) is usually preferred."
        )

    def get_math(self) -> str:
        return r"""
        For mean imputation, a missing value $x_{i,j}$ (where $i$ is the sample, $j$ is the feature)
        is replaced by the mean of the observed values in feature $j$:
        $$ \hat{x}_{i,j} = \mu_j = \frac{1}{N_{obs}} \sum_{k \in \text{observed}} x_{k,j} $$
        """

    def get_code_suggestion(self) -> str:
        return """
        ```python
        from sklearn.impute import SimpleImputer

        # Add SimpleImputer to your pipeline
        pipeline = TutorPipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # strategies: 'mean', 'median', 'most_frequent'
            ('model', YourModelHere())
        ])
        ```
        """

    def get_visual(self, df: pd.DataFrame):
        # Calculate missing values per column
        missing = df.isnull().sum().reset_index()
        missing.columns = ['Feature', 'MissingCount']
        # Filter to show only features with missing values
        missing = missing[missing['MissingCount'] > 0]

        if missing.empty:
            return None

        fig = px.bar(
            missing,
            x='Feature',
            y='MissingCount',
            title="Missing Values Count by Feature",
            labels={'MissingCount': 'Number of Missing Rows'}
        )
        return fig