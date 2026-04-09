"""Pipeline construction utilities for numeric and categorical branches.

This module only assembles sklearn preprocessing pipelines from explicit
configuration.
"""

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def _merge_named_steps(
    base_steps: list[tuple[str, BaseEstimator]],
    override_steps: list[tuple[str, BaseEstimator]],
) -> list[tuple[str, BaseEstimator]]:
    """Merge two named step lists, replacing duplicates while preserving order."""
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
    """Render python source lines for a step list variable."""
    if not steps:
        return [f"{var_name} = []"]

    lines = [f"{var_name} = ["]
    for name, transformer in steps:
        lines.append(f"    ('{name}', {transformer!r}),")
    lines.append("]")
    return lines


def _split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical feature groups."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols


class PipelineBuilder:
    """Build preprocessing branches for numeric and categorical inputs."""

    def __init__(self) -> None:
        """Initialize empty preprocessing step registries."""
        self.numeric_steps: list[tuple[str, BaseEstimator]] = []
        self.categorical_steps: list[tuple[str, BaseEstimator]] = []

    def add_numeric_step(self, name: str, transformer: BaseEstimator) -> None:
        """Add or replace a numeric preprocessing step by name."""
        self.numeric_steps = [(n, t) for n, t in self.numeric_steps if n != name]
        self.numeric_steps.append((name, transformer))

    def add_categorical_step(self, name: str, transformer: BaseEstimator) -> None:
        """Add or replace a categorical preprocessing step by name."""
        self.categorical_steps = [(n, t) for n, t in self.categorical_steps if n != name]
        self.categorical_steps.append((name, transformer))

    def add_step(self, name: str, transformer: BaseEstimator) -> None:
        """Backward-compatible alias that maps to ``add_numeric_step``."""
        self.add_numeric_step(name, transformer)

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer | str:
        """Construct a fitted-ready ``ColumnTransformer`` for the given frame.

        Returns ``'passthrough'`` when no preprocessing steps are configured.
        """
        numeric_cols, categorical_cols = _split_feature_types(X)

        if not self.numeric_steps and not self.categorical_steps:
            return "passthrough"

        transformers: list[tuple[str, BaseEstimator | str, list[str]]] = []

        if numeric_cols:
            if self.numeric_steps:
                numeric_transformer = Pipeline(steps=[(name, clone(step)) for name, step in self.numeric_steps])
            else:
                numeric_transformer = "passthrough"
            transformers.append(("numeric", numeric_transformer, numeric_cols))

        if categorical_cols:
            if self.categorical_steps:
                categorical_transformer = Pipeline(
                    steps=[(name, clone(step)) for name, step in self.categorical_steps]
                )
            else:
                categorical_transformer = "passthrough"
            transformers.append(("categorical", categorical_transformer, categorical_cols))

        if transformers:
            return ColumnTransformer(
                transformers=transformers,
                remainder="passthrough",
                verbose_feature_names_out=False,
            )

        return "passthrough"

    def view_pipeline_code(self, title: str, additional_lines: list[str] | None = None) -> str:
        """Return a readable python code snippet describing the pipeline setup."""
        lines = [
            f"# {title}",
        ]
        if additional_lines:
            lines.extend(additional_lines)

        lines.extend([
            "from sklearn.pipeline import Pipeline",
            "from sklearn.compose import ColumnTransformer",
            "",
        ])
        lines.extend(_render_steps_code("numeric_steps", self.numeric_steps))
        lines.extend(_render_steps_code("categorical_steps", self.categorical_steps))
        lines.extend(
            [
                "",
                "preprocess = ColumnTransformer(transformers=[",
                "    ('numeric', Pipeline(steps=numeric_steps) if numeric_steps else 'passthrough', numeric_cols),",
                "    ('categorical', Pipeline(steps=categorical_steps) if categorical_steps else 'passthrough', categorical_cols),",
                "], remainder='passthrough')",
            ]
        )
        return "\n".join(lines)
