from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from ds_tutor.models import ModelSpec
from ds_tutor.pipeline import PipelineBuilder, _merge_named_steps
from ds_tutor.project import Project

logger = logging.getLogger(__name__)


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
        self._pipeline_builder = PipelineBuilder()
        self._use_project_pipeline = True
        self._pipeline_locked = False

        self._frozen_pipeline_builder: PipelineBuilder | None = None

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

    def _effective_pipeline_builder(self) -> PipelineBuilder:
        if self._pipeline_locked and self._frozen_pipeline_builder is not None:
            return self._frozen_pipeline_builder

        builder = PipelineBuilder()

        if self._use_project_pipeline:
            builder.numeric_steps.extend(
                [(name, clone(step)) for name, step in self.project._pipeline_builder.numeric_steps]
            )
            builder.categorical_steps.extend(
                [(name, clone(step)) for name, step in self.project._pipeline_builder.categorical_steps]
            )

        builder.numeric_steps = _merge_named_steps(builder.numeric_steps, self._pipeline_builder.numeric_steps)
        builder.categorical_steps = _merge_named_steps(
            builder.categorical_steps,
            self._pipeline_builder.categorical_steps,
        )

        return builder

    def _effective_numeric_steps(self) -> list[tuple[str, BaseEstimator]]:
        """Return numeric preprocessing steps after project/experiment merging."""
        return list(self._effective_pipeline_builder().numeric_steps)

    def _effective_categorical_steps(self) -> list[tuple[str, BaseEstimator]]:
        """Return categorical preprocessing steps after project/experiment merging."""
        return list(self._effective_pipeline_builder().categorical_steps)

    def _freeze_pipeline_config(self) -> None:
        self._frozen_pipeline_builder = self._effective_pipeline_builder()
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
        self._pipeline_builder = PipelineBuilder()

    def add_numeric_step(self, name: str, transformer: BaseEstimator) -> None:
        self._assert_pipeline_mutable()
        self._pipeline_builder.add_numeric_step(name, transformer)

    def add_categorical_step(self, name: str, transformer: BaseEstimator) -> None:
        self._assert_pipeline_mutable()
        self._pipeline_builder.add_categorical_step(name, transformer)

    def add_step(self, name: str, transformer: BaseEstimator) -> None:
        self._assert_pipeline_mutable()
        self._pipeline_builder.add_step(name, transformer)

    def add_theory(self, message: str) -> None:
        self.theory_log.append(message)

    def _build_pipeline(self, estimator: BaseEstimator, X: pd.DataFrame) -> Pipeline:
        builder = self._effective_pipeline_builder()
        preprocessor = builder.build_preprocessor(X)
        return Pipeline(steps=[("preprocess", preprocessor), ("model", clone(estimator))])

    def view_pipeline(self) -> str:
        builder = self._effective_pipeline_builder()
        additional_lines = [
            f"use_project_pipeline = {self._use_project_pipeline}",
            f"pipeline_locked = {self._pipeline_locked}",
        ]
        code = builder.view_pipeline_code(f"Experiment pipeline: {self.name}", additional_lines)
        code += "\npipeline = Pipeline(steps=[('preprocess', preprocess), ('model', estimator)])"
        print(code)
        return code

    def run(self) -> "Experiment":
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
            pipeline = self._build_pipeline(model_spec.estimator, X_train)

            param_grid = {f"model__{k}": v for k, v in (model_spec.param_grid or {}).items()}

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
            search.fit(X_train, y_train)

            for i, params in enumerate(search.cv_results_["params"]):
                original_params = {k.replace("model__", ""): v for k, v in params.items()}
                mean_score = float(search.cv_results_["mean_test_score"][i])

                self._results.append(
                    {
                        "model": model_spec.name,
                        "params": original_params,
                        "mean_cv_score": mean_score,
                        "std_cv_score": float(search.cv_results_["std_test_score"][i]),
                    }
                )

            if search.best_score_ > best_score:
                best_score = float(search.best_score_)
                best_pipeline = search.best_estimator_
                best_model_name = model_spec.name
                best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

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
