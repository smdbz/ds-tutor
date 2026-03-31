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
    """The immutable environment. Enforces causality and prevents data leakage."""

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


class ExperimentConfig:
    """The declarative state machine."""

    def __init__(self, name: str, project_context: ProjectContext):
        self.name = name
        self.project_context = project_context
        self.preprocessing_steps = []
        self.estimator = None
        self.is_locked = False
        self.score = None

    def add_step(self, step_name: str, step_obj):
        if self.is_locked: raise RuntimeError("Config is locked!")
        self.preprocessing_steps.append((step_name, step_obj))

    def set_estimator(self, estimator_name: str, estimator_obj):
        if self.is_locked: raise RuntimeError("Config is locked!")
        self.estimator = (estimator_name, estimator_obj)

    def lock(self, score: float):
        self.score = score
        self.is_locked = True
        print(f"[SYSTEM] Experiment '{self.name}' LOCKED with CV MSE: {self.score:.4f}")


# =====================================================================
# 2. THE PEDAGOGICAL TUTOR (THE "SECOND BRAIN")
# =====================================================================

class ExploratoryDataAnalysisTutor:
    """Diagnoses and visualizes the data."""

    def __init__(self, context: ProjectContext, config: ExperimentConfig):
        self.context = context
        self.config = config
        self._tutors: list = []

    def add_tutor(self, tutor, **kwargs):
        tutor = tutor(self.context, self.config, **kwargs)
        self._tutors.append(tutor)

    def teach_me(self):
        for tutor in self._tutors:
            tutor.teach_me()


class MissingDataTutor:
    """Diagnoses missingness and declaratively updates the config."""

    def __init__(self, context: ProjectContext, config: ExperimentConfig):
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

    def __init__(self, context: ProjectContext, config: ExperimentConfig, threshold: float = 1.0):
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


# =====================================================================
# 3. THE EXECUTION ENGINE
# =====================================================================

class PipelineRunner:
    """Translates the declarative config into a scikit-learn DAG and executes."""

    def execute(self, context: ProjectContext, config: ExperimentConfig):
        print(f"[SYSTEM] Building pipeline for '{config.name}'...")
        X_train, y_train = context.training_data

        # Build the DAG from the config
        steps = config.preprocessing_steps.copy()
        if config.estimator is None:
            raise ValueError("No estimator set in config!")
        steps.append(config.estimator)

        pipeline = Pipeline(steps)

        # Execute cross-validation (this proves no leakage occurs during imputation)
        print("[SYSTEM] Running 5-Fold Cross Validation...")
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mse = -scores.mean()

        config.lock(mse)
