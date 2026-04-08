from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer

from ds_tutor.core import Experiment, Project, detect_text_columns
from ds_tutor.theory import load_theory


@dataclass(frozen=True)
class TutorResult:
    tutor_name: str
    title: str
    triggered: bool
    diagnosis: str
    action: str
    theory: str
    details: pd.DataFrame | None = None


class SupportsTeachMe(Protocol):
    def teach_me(self) -> TutorResult: ...


class ExploratoryDataAnalysisTutor:
    """Runs a sequence of tutors that can modify experiment configuration."""

    def __init__(self, context: Project, config: Experiment):
        self.context = context
        self.config = config
        self._tutors: list[SupportsTeachMe] = [
            MissingDataTutor(self.context, self.config),
            SkewnessTutor(self.context, self.config),
            TextDataTutor(self.context, self.config),
        ]
        self._last_results: list[TutorResult] = []

    def add_tutor(self, tutor_cls, **kwargs) -> None:
        self._tutors.append(tutor_cls(self.context, self.config, **kwargs))

    def list_tutors(self) -> list[str]:
        return [type(t).__name__ for t in self._tutors]

    @property
    def last_results(self) -> list[TutorResult]:
        return list(self._last_results)

    def teach_me(self) -> list[TutorResult]:
        results: list[TutorResult] = []
        for tutor in self._tutors:
            results.append(tutor.teach_me())
        self._last_results = results
        return results

    def summary(self) -> None:
        """Render a dedicated EDA tutor summary in Jupyter."""
        try:
            from ds_tutor.ui.jupyter import render_eda_tutor_summary
        except ImportError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            print(
                "Jupyter UI dependencies are not available for this kernel. "
                f"Missing import: {missing}. "
                "Install with: pip install ipython ipywidgets rich"
            )
            return
        render_eda_tutor_summary(self)


class MissingDataTutor:
    """Adds an imputer step when missing numeric values are detected."""

    def __init__(self, context: Project, config: Experiment):
        self.context = context
        self.config = config

    def teach_me(self) -> TutorResult:
        X_train, _ = self.context.training_data
        numeric = X_train.select_dtypes(include="number")
        missing_by_feature = numeric.isnull().sum().sort_values(ascending=False)
        missing_by_feature = missing_by_feature[missing_by_feature > 0]
        missing_count = int(missing_by_feature.sum())
        theory = load_theory("missing_data")

        if missing_count <= 0:
            return TutorResult(
                tutor_name=type(self).__name__,
                title="Missing Data",
                triggered=False,
                diagnosis="No missing numeric values detected in training data.",
                action="No preprocessing step added.",
                theory=theory,
                details=None,
            )

        self.config.add_step("imputer", SimpleImputer(strategy="median"))
        self.config.add_theory(
            "MissingDataTutor: Added numeric median imputer because missing values were detected."
        )
        self.config.add_theory(theory)
        total_rows = max(1, len(numeric))
        details = pd.DataFrame(
            {
                "feature": missing_by_feature.index,
                "missing_count": missing_by_feature.values,
                "missing_pct": (missing_by_feature.values / total_rows) * 100.0,
            }
        )
        return TutorResult(
            tutor_name=type(self).__name__,
            title="Missing Data",
            triggered=True,
            diagnosis=f"Detected {missing_count} missing numeric values across {len(details)} feature(s).",
            action="Added numeric median imputer (`SimpleImputer(strategy='median')`).",
            theory=theory,
            details=details,
        )


class SkewnessTutor:
    """Adds a power transform step when highly skewed numeric features are detected."""

    def __init__(self, context: Project, config: Experiment, threshold: float = 1.0):
        self.context = context
        self.config = config
        self.threshold = threshold

    def teach_me(self) -> TutorResult:
        X_train, _ = self.context.training_data
        numeric = X_train.select_dtypes(include="number")
        theory = load_theory("skewness")
        if numeric.empty:
            return TutorResult(
                tutor_name=type(self).__name__,
                title="Skewness",
                triggered=False,
                diagnosis="No numeric features found.",
                action="No preprocessing step added.",
                theory=theory,
                details=None,
            )

        skew = numeric.skew().abs()
        skewed = skew[skew > self.threshold].sort_values(ascending=False)
        if skewed.empty:
            return TutorResult(
                tutor_name=type(self).__name__,
                title="Skewness",
                triggered=False,
                diagnosis=f"No features exceeded |skew| > {self.threshold}.",
                action="No preprocessing step added.",
                theory=theory,
                details=None,
            )

        self.config.add_step("power_transformer", PowerTransformer(method="yeo-johnson"))
        summary = ", ".join(f"{col}={value:.3f}" for col, value in skewed.items())
        self.config.add_theory(
            f"SkewnessTutor: Added Yeo-Johnson power transform for skewed features ({summary})."
        )
        self.config.add_theory(theory)
        details = pd.DataFrame({"feature": skewed.index, "abs_skew": skewed.values})
        return TutorResult(
            tutor_name=type(self).__name__,
            title="Skewness",
            triggered=True,
            diagnosis=f"{len(details)} feature(s) exceeded |skew| > {self.threshold}.",
            action="Added Yeo-Johnson power transform (`PowerTransformer(method='yeo-johnson')`).",
            theory=theory,
            details=details,
        )


class TextDataTutor:
    """Adds TF-IDF preprocessing when free-form text features are detected."""

    def __init__(
        self,
        context: Project,
        config: Experiment,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        self.context = context
        self.config = config
        self.max_features = max_features
        self.ngram_range = ngram_range

    def teach_me(self) -> TutorResult:
        X_train, _ = self.context.training_data
        text_cols = detect_text_columns(X_train)
        theory = load_theory("text_data")

        if not text_cols:
            return TutorResult(
                tutor_name=type(self).__name__,
                title="Text Data",
                triggered=False,
                diagnosis="No free-form text columns detected.",
                action="No text preprocessing step added.",
                theory=theory,
                details=None,
            )

        self.config.add_text_step(
            "tfidf",
            TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range),
        )
        self.config.add_theory(
            "TextDataTutor: Added TF-IDF vectorization for detected text columns."
        )
        self.config.add_theory(theory)

        details = pd.DataFrame(
            {
                "feature": text_cols,
                "avg_char_length": [
                    float(X_train[col].dropna().astype(str).str.len().mean() or 0.0) for col in text_cols
                ],
            }
        )

        return TutorResult(
            tutor_name=type(self).__name__,
            title="Text Data",
            triggered=True,
            diagnosis=f"Detected {len(text_cols)} text feature(s): {', '.join(text_cols)}.",
            action=(
                "Added text TF-IDF step "
                f"(`TfidfVectorizer(max_features={self.max_features}, ngram_range={self.ngram_range})`)."
            ),
            theory=theory,
            details=details,
        )
