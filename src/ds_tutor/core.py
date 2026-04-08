from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import ParameterGrid, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    param_grid: dict[str, list[Any]]


def _merge_text_columns(frame: pd.DataFrame) -> pd.Series:
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    return frame.fillna("").astype(str).agg(" ".join, axis=1)


def _is_text_series(series: pd.Series) -> bool:
    is_categorical = isinstance(series.dtype, pd.CategoricalDtype)
    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) or is_categorical):
        return False

    sample = series.dropna().astype(str)
    if sample.empty:
        return False

    avg_words = float(sample.str.split().map(len).mean())
    avg_length = float(sample.str.len().mean())
    unique_ratio = float(sample.nunique() / max(1, len(sample)))
    return avg_words >= 2.0 or (avg_length >= 20.0 and unique_ratio >= 0.2)


def detect_text_columns(frame: pd.DataFrame, candidate_cols: list[str] | None = None) -> list[str]:
    cols = candidate_cols or frame.columns.tolist()
    return [col for col in cols if _is_text_series(frame[col])]


def _merge_named_steps(base_steps, override_steps):
    merged = list(base_steps)
    override_dict = dict(override_steps)

    # Replace existing steps in-place to preserve order
    for i, (name, _) in enumerate(merged):
        if name in override_dict:
            merged[i] = (name, override_dict.pop(name))

    # Append any entirely new steps to the end
    merged.extend(override_dict.items())
    return merged


def _render_steps_code(var_name: str, steps: list[tuple[str, BaseEstimator]]) -> list[str]:
    if not steps:
        return [f"{var_name} = []"]

    lines = [f"{var_name} = ["]
    for name, transformer in steps:
        lines.append(f"    ('{name}', {transformer!r}),")
    lines.append("]")
    return lines


class Project:
    """Immutable dataset context plus default pipeline configuration for experiments."""

    def __init__(
        self,
        name: str,
        target_col: str,
        df: pd.DataFrame,
        test: pd.DataFrame | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.name = name
        self.target_col = target_col
        self.competition_mode = test is not None
        self._experiments: list[Experiment] = []

        self._numeric_steps: list[tuple[str, BaseEstimator]] = []
        self._categorical_steps: list[tuple[str, BaseEstimator]] = []
        self._text_steps: list[tuple[str, BaseEstimator]] = []

        if test is None:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if y.nunique() < max(2, int(len(y) * test_size)) else None,
            )
        else:
            self._X_train = df.drop(columns=[target_col])
            self._y_train = df[target_col]
            self._X_test = test
            self._y_test = None

    @property
    def training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._X_train.copy(), self._y_train.copy()

    @property
    def test_data(self) -> tuple[pd.DataFrame, pd.Series | None]:
        y = None if self._y_test is None else self._y_test.copy()
        return self._X_test.copy(), y

    def _register_experiment(self, experiment: "Experiment") -> None:
        self._experiments.append(experiment)

    def add_numeric_step(self, name: str, transformer: BaseEstimator) -> None:
        self._numeric_steps = [(n, t) for n, t in self._numeric_steps if n != name]
        self._numeric_steps.append((name, transformer))

    def add_categorical_step(self, name: str, transformer: BaseEstimator) -> None:
        self._categorical_steps = [(n, t) for n, t in self._categorical_steps if n != name]
        self._categorical_steps.append((name, transformer))

    def add_text_step(self, name: str, transformer: BaseEstimator) -> None:
        self._text_steps = [(n, t) for n, t in self._text_steps if n != name]
        self._text_steps.append((name, transformer))

    def view_pipeline(self) -> str:
        lines = [
            f"# Project pipeline defaults: {self.name}",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.compose import ColumnTransformer",
            "from sklearn.preprocessing import FunctionTransformer",
            "",
        ]
        lines.extend(_render_steps_code("numeric_steps", self._numeric_steps))
        lines.extend(_render_steps_code("categorical_steps", self._categorical_steps))
        lines.extend(_render_steps_code("text_steps", self._text_steps))
        lines.extend(
            [
                "",
                "preprocess = ColumnTransformer(transformers=[",
                "    ('numeric', Pipeline(steps=numeric_steps) if numeric_steps else 'passthrough', numeric_cols),",
                "    ('categorical', Pipeline(steps=categorical_steps) if categorical_steps else 'passthrough', categorical_cols),",
                "    ('text', Pipeline(steps=[('combine_text', FunctionTransformer(_merge_text_columns, validate=False)), *text_steps]) if text_steps else 'passthrough', text_cols),",
                "], remainder='passthrough')",
            ]
        )

        code = "\n".join(lines)
        print(code)
        return code

    def leaderboard(self) -> pd.DataFrame:
        rows = []
        for exp in self._experiments:
            if exp.best_score is None:
                continue
            rows.append(
                {
                    "experiment": exp.name,
                    "best_model": exp.best_model_name,
                    "best_cv_score": exp.best_score,
                    "holdout_score": exp.holdout_score,
                }
            )
        if not rows:
            return pd.DataFrame(columns=["experiment", "best_model", "best_cv_score", "holdout_score"])
        return pd.DataFrame(rows).sort_values("best_cv_score", ascending=False, ignore_index=True)


class Experiment:
    """Searches over models/hyperparameters and stores run artifacts plus pipeline config."""

    def __init__(
        self,
        name: str,
        project: Project,
        cv: int = 5,
        scoring: str | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.name = name
        self.project = project
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

        self._models: list[ModelSpec] = []
        self._numeric_steps: list[tuple[str, BaseEstimator]] = []
        self._categorical_steps: list[tuple[str, BaseEstimator]] = []
        self._text_steps: list[tuple[str, BaseEstimator]] = []
        self._use_project_pipeline = True
        self._pipeline_locked = False

        self._frozen_numeric_steps: list[tuple[str, BaseEstimator]] | None = None
        self._frozen_categorical_steps: list[tuple[str, BaseEstimator]] | None = None
        self._frozen_text_steps: list[tuple[str, BaseEstimator]] | None = None

        self._results: list[dict[str, Any]] = []

        self.best_pipeline: Pipeline | None = None
        self.best_model_name: str | None = None
        self.best_params: dict[str, Any] | None = None
        self.best_score: float | None = None
        self.holdout_score: float | None = None
        self.theory_log: list[str] = []

        self.project._register_experiment(self)

    def _assert_pipeline_mutable(self) -> None:
        if self._pipeline_locked:
            raise RuntimeError("Pipeline is immutable after the experiment has run once.")

    def _effective_numeric_steps(self) -> list[tuple[str, BaseEstimator]]:
        if self._pipeline_locked and self._frozen_numeric_steps is not None:
            return list(self._frozen_numeric_steps)
        base = self.project._numeric_steps if self._use_project_pipeline else []
        return _merge_named_steps(base, self._numeric_steps)

    def _effective_categorical_steps(self) -> list[tuple[str, BaseEstimator]]:
        if self._pipeline_locked and self._frozen_categorical_steps is not None:
            return list(self._frozen_categorical_steps)
        base = self.project._categorical_steps if self._use_project_pipeline else []
        return _merge_named_steps(base, self._categorical_steps)

    def _effective_text_steps(self) -> list[tuple[str, BaseEstimator]]:
        if self._pipeline_locked and self._frozen_text_steps is not None:
            return list(self._frozen_text_steps)
        base = self.project._text_steps if self._use_project_pipeline else []
        return _merge_named_steps(base, self._text_steps)

    def _freeze_pipeline_config(self) -> None:
        self._frozen_numeric_steps = [(name, clone(step)) for name, step in self._effective_numeric_steps()]
        self._frozen_categorical_steps = [(name, clone(step)) for name, step in self._effective_categorical_steps()]
        self._frozen_text_steps = [(name, clone(step)) for name, step in self._effective_text_steps()]
        self._pipeline_locked = True

    def add_model(
        self,
        name: str,
        estimator: BaseEstimator,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> None:
        self._models.append(ModelSpec(name=name, estimator=estimator, param_grid=param_grid or {}))

    def use_project_pipeline(self, enabled: bool = True) -> None:
        self._assert_pipeline_mutable()
        self._use_project_pipeline = enabled

    def override_pipeline(self) -> None:
        self._assert_pipeline_mutable()
        self._use_project_pipeline = False
        self._numeric_steps = []
        self._categorical_steps = []
        self._text_steps = []

    def add_numeric_step(self, name: str, transformer: BaseEstimator) -> None:
        self._assert_pipeline_mutable()
        self._numeric_steps = [(n, t) for n, t in self._numeric_steps if n != name]
        self._numeric_steps.append((name, transformer))

    def add_categorical_step(self, name: str, transformer: BaseEstimator) -> None:
        self._assert_pipeline_mutable()
        self._categorical_steps = [(n, t) for n, t in self._categorical_steps if n != name]
        self._categorical_steps.append((name, transformer))

    def add_text_step(self, name: str, transformer: BaseEstimator) -> None:
        self._assert_pipeline_mutable()
        self._text_steps = [(n, t) for n, t in self._text_steps if n != name]
        self._text_steps.append((name, transformer))

    def add_theory(self, message: str) -> None:
        self.theory_log.append(message)

    def _split_feature_types(
        self,
        X: pd.DataFrame,
    ) -> tuple[list[str], list[str], list[str]]:
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        non_numeric = X.select_dtypes(exclude="number").columns.tolist()
        text_cols = detect_text_columns(X, non_numeric)
        categorical_cols = [col for col in non_numeric if col not in text_cols]
        return numeric_cols, categorical_cols, text_cols

    def _build_pipeline(self, estimator: BaseEstimator, X: pd.DataFrame) -> Pipeline:
        numeric_steps = self._effective_numeric_steps()
        categorical_steps = self._effective_categorical_steps()
        text_steps = self._effective_text_steps()

        numeric_cols, categorical_cols, text_cols = self._split_feature_types(X)

        if not numeric_steps and not categorical_steps and not text_steps:
            preprocessor: ColumnTransformer | str = "passthrough"
        else:
            transformers: list[tuple[str, BaseEstimator | str, list[str]]] = []

            if numeric_cols:
                numeric_transformer: BaseEstimator | str = "passthrough"
                if numeric_steps:
                    numeric_transformer = Pipeline(steps=[(name, clone(step)) for name, step in numeric_steps])
                transformers.append(("numeric", numeric_transformer, numeric_cols))

            if categorical_cols:
                categorical_transformer: BaseEstimator | str = "passthrough"
                if categorical_steps:
                    categorical_transformer = Pipeline(
                        steps=[(name, clone(step)) for name, step in categorical_steps]
                    )
                transformers.append(("categorical", categorical_transformer, categorical_cols))

            if text_cols:
                text_transformer: BaseEstimator | str = "passthrough"
                if text_steps:
                    text_transformer = Pipeline(
                        steps=[
                            ("combine_text", FunctionTransformer(_merge_text_columns, validate=False)),
                            *[(name, clone(step)) for name, step in text_steps],
                        ]
                    )
                transformers.append(("text", text_transformer, text_cols))

            if transformers:
                preprocessor = ColumnTransformer(
                    transformers=transformers,
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                )
            else:
                preprocessor = "passthrough"

        return Pipeline(steps=[("preprocess", preprocessor), ("model", clone(estimator))])

    def view_pipeline(self) -> str:
        numeric_steps = self._effective_numeric_steps()
        categorical_steps = self._effective_categorical_steps()
        text_steps = self._effective_text_steps()

        lines = [
            f"# Experiment pipeline: {self.name}",
            f"use_project_pipeline = {self._use_project_pipeline}",
            f"pipeline_locked = {self._pipeline_locked}",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.compose import ColumnTransformer",
            "from sklearn.preprocessing import FunctionTransformer",
            "",
        ]
        lines.extend(_render_steps_code("numeric_steps", numeric_steps))
        lines.extend(_render_steps_code("categorical_steps", categorical_steps))
        lines.extend(_render_steps_code("text_steps", text_steps))
        lines.extend(
            [
                "",
                "preprocess = ColumnTransformer(transformers=[",
                "    ('numeric', Pipeline(steps=numeric_steps) if numeric_steps else 'passthrough', numeric_cols),",
                "    ('categorical', Pipeline(steps=categorical_steps) if categorical_steps else 'passthrough', categorical_cols),",
                "    ('text', Pipeline(steps=[('combine_text', FunctionTransformer(_merge_text_columns, validate=False)), *text_steps]) if text_steps else 'passthrough', text_cols),",
                "], remainder='passthrough')",
                "pipeline = Pipeline(steps=[('preprocess', preprocess), ('model', estimator)])",
            ]
        )

        code = "\n".join(lines)
        print(code)
        return code

    def run(self) -> "Experiment":
        from sklearn.model_selection import GridSearchCV

        if not self._models:
            raise ValueError("No models configured. Use add_model() before run().")

        X_train, y_train = self.project.training_data
        X_test, y_test = self.project.test_data

        self._results = []
        best_score = float("-inf")
        best_pipeline: Pipeline | None = None
        best_model_name: str | None = None
        best_params: dict[str, Any] | None = None

        for model_spec in self._models:
            # Build the base pipeline ONCE per model
            pipeline = self._build_pipeline(model_spec.estimator, X_train)

            # Map parameters to the 'model' step in the pipeline
            param_grid = {f"model__{k}": v for k, v in (model_spec.param_grid or {}).items()}

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
            search.fit(X_train, y_train)

            # Automatically extract the results and standard deviations
            for i, params in enumerate(search.cv_results_["params"]):
                # Remove the 'model__' prefix for clean logging
                original_params = {k.replace("model__", ""): v for k, v in params.items()}
                self._results.append({
                    "model": model_spec.name,
                    "params": original_params,
                    "mean_cv_score": float(search.cv_results_["mean_test_score"][i]),
                    "std_cv_score": float(search.cv_results_["std_test_score"][i]),
                })

        assert best_pipeline is not None
        self.best_pipeline = best_pipeline.fit(X_train, y_train)
        self.best_model_name = best_model_name
        self.best_params = best_params
        self.best_score = best_score

        if y_test is not None:
            self.holdout_score = float(self.best_pipeline.score(X_test, y_test))
        else:
            self.holdout_score = None

        if not self._pipeline_locked:
            self._freeze_pipeline_config()

        return self

    def leaderboard(self, top_n: int | None = None) -> pd.DataFrame:
        if not self._results:
            return pd.DataFrame(columns=["model", "params", "mean_cv_score", "std_cv_score"])

        df = pd.DataFrame(self._results).sort_values("mean_cv_score", ascending=False, ignore_index=True)
        if top_n is not None:
            return df.head(top_n).reset_index(drop=True)
        return df

    def predict(self, X: pd.DataFrame | None = None) -> pd.Series:
        if self.best_pipeline is None:
            raise ValueError("Experiment has not been run. Call run() first.")

        if X is None:
            X, _ = self.project.test_data

        preds = self.best_pipeline.predict(X)
        return pd.Series(preds, index=X.index, name=f"{self.name}_prediction")

    def summary(self) -> None:
        """Render an interactive summary in Jupyter if available."""
        try:
            from ds_tutor.ui.jupyter import render_experiment_summary
        except ImportError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            logger.error(
                "Jupyter UI dependencies are not available for this kernel. "
                "Missing import: %s. Install with: pip install ipython ipywidgets rich",
                missing,
            )
            return

        render_experiment_summary(self)
