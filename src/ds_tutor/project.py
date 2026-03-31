from dataclasses import dataclass, field
from importlib.resources import files

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


def load_theory(name: str) -> str:
    return files("ds_tutor.notes").joinpath(f"{name}.md").read_text()


# =====================================================================
# 1. CORE ML-OPS ARCHITECTURE
# =====================================================================

class ProjectContext:
    """The immutable environment. Enforces causality and prevents data leakage. A project is one dataset that will be
     experimented with."""

    def __init__(self, name: str, df: pd.DataFrame, target_col: str):
        self.name = name
        self.raw_data = df.copy()
        self.target_col = target_col
        self._is_initialized = False

    def initialize(self):
        """
        Drop the target column and split the data into train/test sets.
        :return:
        """
        X = self.raw_data.drop(columns=[self.target_col])
        y = self.raw_data[self.target_col]
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.expected_value = self._y_train.mean()
        self._is_initialized = True

        print(f"[SYSTEM] Context Initialized. Target leakage prevented.")
        print(f"[MATH] Naive Baseline E[Y] = {self.expected_value:.2f}\n")

    @property
    def training_data(self):
        if not self._is_initialized:
            raise RuntimeError("Must call initialize() first!")
        return self._X_train.copy(), self._y_train.copy()


@dataclass
class ExperimentResults:
    """The results of a completed experiment."""
    cv_scores: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    baseline: float = 0.0

    def __repr__(self):
        return (
            f"ExperimentResults(mean={self.mean:.4f}, std={self.std:.4f}, "
            f"baseline={self.baseline:.4f}, folds={len(self.cv_scores)})"
        )


class Experiment:
    """An experiment represents one attempt at training a model. It has a hypothesis, a configuration
    (preprocessing steps + estimator), and produces results via cross-validation."""

    def __init__(self, name: str, project_context: ProjectContext, hypothesis: str = ""):
        self.name = name
        self.project_context = project_context
        self.hypothesis = hypothesis
        self.preprocessing_steps = []
        self.estimator = None
        self.is_locked = False
        self.results: ExperimentResults | None = None

    def add_step(self, step_name: str, step_obj):
        if self.is_locked: raise RuntimeError(
            "Experiment is locked! Create a new experiment to try a different approach.")
        self.preprocessing_steps.append((step_name, step_obj))

    def set_estimator(self, estimator_obj):
        if self.is_locked: raise RuntimeError(
            "Experiment is locked! Create a new experiment to try a different approach.")
        self.estimator = estimator_obj

    def run(self, cv: int = 5, scoring: str = "neg_mean_squared_error"):
        """Run the experiment: build the pipeline and execute cross-validation."""
        if self.is_locked:
            raise RuntimeError("Experiment already ran! Create a new experiment to try a different approach.")
        if self.estimator is None:
            raise ValueError("No estimator set! Call set_estimator() before running.")

        print(f"[EXPERIMENT] Running '{self.name}'...")
        if self.hypothesis:
            print(f"[HYPOTHESIS] {self.hypothesis}")

        X_train, y_train = self.project_context.training_data

        steps = self.preprocessing_steps.copy()
        steps.append(self.estimator)
        pipeline = Pipeline(steps)

        print(f"[SYSTEM] Running {cv}-Fold Cross Validation...")
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        cv_scores = (-scores).tolist()

        self.results = ExperimentResults(
            cv_scores=cv_scores,
            mean=sum(cv_scores) / len(cv_scores),
            std=float(pd.Series(cv_scores).std()),
            baseline=self.project_context.expected_value,
        )
        self._finalize()

    def _finalize(self):
        self.is_locked = True
        print(
            f"[RESULT] Experiment '{self.name}' complete — CV MSE: {self.results.mean:.4f} (+/- {self.results.std:.4f})")
        print(f"[RESULT] Naive baseline E[Y]: {self.results.baseline:.2f}")


# =====================================================================
# 2. THE PEDAGOGICAL TUTOR (THE "SECOND BRAIN")
# =====================================================================

class ExploratoryDataAnalysisTutor:
    """Diagnoses and visualizes the data."""

    def __init__(self, context: ProjectContext, config: Experiment):
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

    def __init__(self, context: ProjectContext, config: Experiment):
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

    def __init__(self, context: ProjectContext, config: Experiment, threshold: float = 1.0):
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


