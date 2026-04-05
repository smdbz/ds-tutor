import logging
from importlib.resources import files

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

logger = logging.getLogger(__name__)


def load_theory(name: str) -> str:
    return files("ds_tutor.notes").joinpath(f"{name}.md").read_text()


# =====================================================================
# 1. CORE ML-OPS ARCHITECTURE
# =====================================================================

class Project:
    """The immutable environment. Enforces causality and prevents data leakage. A project is one dataset that will be
     experimented with."""

    def __init__(self, name: str, df: pd.DataFrame, target_col: str):
        self.name = name
        self.raw_data = df.copy()
        self.target_col = target_col
        X = self.raw_data.drop(columns=[self.target_col])
        y = self.raw_data[self.target_col]
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.expected_value = self._y_train.mean()
        # 2. Use the local logger, not the root logging object
        logger.info(f"Project '{self.name}' loaded. Training set shape: {self._X_train.shape}")
        logger.debug(f"Naive Baseline E[Y] established at: {self.expected_value:.4f}")

    @property
    def training_data(self):
        return self._X_train.copy(), self._y_train.copy()

    def add_step(self):
        """All trials in the project will share this step."""
        pass


class Experiment:
    """An Experiment contains one or more Trials."""

    def __init__(self, name: str, project: Project):
        self.name = name
        self.project = project

    def leaderboard(self):
        pass

    def add_step(self):
        """All trials in the experiment will share this step."""
        pass

    def add_trial(self, trial: Trial):
        pass


class Trial:
    """A trial represents one run of a cross-validation pipeline. Trial has a score and can be compared to other trials."""

    def __init__(self, name: str, project: Project, experiment: Experiment):
        self.name = name
        self.project = project
        self.experiment = experiment
        self.score = None

        self.numeric_transformers: list[tuple[str, BaseEstimator]] = []
        self.categorical_transformers: list[tuple[str, BaseEstimator]] = []
        self.preprocessor = ColumnTransformer(transformers=[self.numeric_transformers + self.categorical_transformers])

        self.pipeline_steps = []
        self.pipeline = Pipeline(steps=self.pipeline_steps)  # empty pipeline

    def add_numeric_transformer(self, name: str, transformer: BaseEstimator):
        self.numeric_transformers.append((name, transformer))

    def preprocessor(self):
        pass

    def get_data(self):
        X, y = self.project.training_data

    def add_step(self):
        pass

    def build_pipeline(self, ):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def score(self):
        pass

    def cross_validate(self):
        pass

    def report(self):
        pass


# =====================================================================
# 2. THE PEDAGOGICAL TUTOR (THE "SECOND BRAIN")
# =====================================================================

class ExploratoryDataAnalysisTutor:
    """Diagnoses and visualizes the data."""

    def __init__(self, context: Project, config: Experiment):
        self.context = context
        self.config = config
        self._tutors: list = [MissingDataTutor(self.context, self.config), SkewnessTutor(self.context, self.config)]

    def add_tutor(self, tutor, **kwargs):
        tutor = tutor(self.context, self.config, **kwargs)
        self._tutors.append(tutor)

    def list_tutors(self):
        for tutor in self._tutors:
            print(tutor)

    def teach_me(self):
        for tutor in self._tutors:
            tutor.teach_me()


class MissingDataTutor:
    """Diagnoses missingness and declaratively updates the config."""

    def __init__(self, context: Project, config: Experiment):
        self.context = context
        self.config = config

    def teach_me(self):
        X_train, _ = self.context.training_data
        missing_count = X_train.isnull().sum().sum()

        if missing_count > 0:
            print("=" * 60)
            print("[TUTOR TRIGGERED] : Missing Data Detected")
            print("=" * 60)
            print(f"[DIAGNOSIS] Found {missing_count} missing values in training data.")
            print(f"[THEORY] {load_theory('missing_data')}")
            print("[ACTION] Adding Median SimpleImputer to the declarative configuration.\n")

            # Mutate the configuration, not the dataframe!
            self.config.add_step("imputer", SimpleImputer(strategy="median"))
        else:
            print("[TUTOR] No missing data detected. Passing.")

    def print_pandas(self):
        print("X_train.isnull().sum().sum()")


class SkewnessTutor:
    """Diagnoses skewed numeric features and declaratively updates the config."""

    def __init__(self, context: Project, config: Experiment, threshold: float = 1.0):
        self.context = context
        self.config = config
        self.threshold = threshold

    def teach_me(self):
        X_train, _ = self.context.training_data
        numeric_cols = X_train.select_dtypes(include="number").columns
        skew = X_train[numeric_cols].skew().abs()
        skewed_cols = skew[skew > self.threshold].sort_values(ascending=False)

        if not skewed_cols.empty:
            print("=" * 60)
            print("[TUTOR TRIGGERED] : Skewed Features Detected")
            print("=" * 60)
            print(f"[DIAGNOSIS] {len(skewed_cols)} feature(s) exceed |skew| > {self.threshold}:")
            for col, val in skewed_cols.items():
                print(f"  {col}: skew = {val:.3f}")
            print()
            print(f"[THEORY] {load_theory('skewness')}")
            print()
            print("[GOTCHA] PowerTransformer requires all values to be finite.")
            print("  Run MissingDataTutor BEFORE SkewnessDataTutor so the")
            print("  imputer is added to the pipeline first.")
            print()
            print("[ACTION] Adding Yeo-Johnson PowerTransformer to the declarative configuration.\n")

            # Mutate the configuration, not the dataframe!
            self.config.add_step("power_transformer", PowerTransformer(method="yeo-johnson"))
        else:
            print(f"[TUTOR] No features exceed |skew| > {self.threshold}. Passing.")

    def print_pandas(self):
        print("X_train.select_dtypes(include='number').skew().abs()")
