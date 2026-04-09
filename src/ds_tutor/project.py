from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from ds_tutor.pipeline import PipelineBuilder

if TYPE_CHECKING:
    from ds_tutor.experiment import Experiment


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
        self._experiments: list["Experiment"] = []

        self._pipeline_builder = PipelineBuilder()

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
        self._pipeline_builder.add_numeric_step(name, transformer)

    def add_categorical_step(self, name: str, transformer: BaseEstimator) -> None:
        self._pipeline_builder.add_categorical_step(name, transformer)

    def add_step(self, name: str, transformer: BaseEstimator) -> None:
        self._pipeline_builder.add_step(name, transformer)

    def view_pipeline(self) -> str:
        code = self._pipeline_builder.view_pipeline_code(f"Project pipeline defaults: {self.name}")
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
