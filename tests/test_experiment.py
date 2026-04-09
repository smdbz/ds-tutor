import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer

from ds_tutor import Experiment, ExploratoryDataAnalysisTutor, Project


def _classification_df(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, rows)
    x2 = rng.normal(2, 1.5, rows)
    logits = 0.9 * x1 - 0.3 * x2
    y = (logits > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_experiment_runs_multi_model_search_and_stores_results():
    df = _classification_df()
    project = Project(name="demo", target_col="target", df=df)

    experiment = Experiment(name="search", project=project, cv=3)
    experiment.add_model("logreg", LogisticRegression(max_iter=2000), {"C": [0.1, 1.0]})
    experiment.add_model("rf", RandomForestClassifier(random_state=42), {"n_estimators": [10, 30]})

    experiment.run()

    leaderboard = experiment.leaderboard()
    assert not leaderboard.empty
    assert experiment.best_pipeline is not None
    assert experiment.best_model_name in {"logreg", "rf"}
    assert experiment.best_score is not None
    assert project.leaderboard().shape[0] == 1


def test_tutors_append_preprocessing_steps_and_theory_log():
    df = _classification_df()
    df.loc[df.index[:8], "x1"] = np.nan
    df["x2"] = np.exp(df["x2"])  # create skew

    project = Project(name="demo", target_col="target", df=df)
    experiment = Experiment(name="search", project=project, cv=3)
    experiment.add_model("logreg", LogisticRegression(max_iter=2000), {"C": [1.0]})

    tutor = ExploratoryDataAnalysisTutor(project, experiment)
    results = tutor.teach_me()
    experiment.run()

    assert len(results) == 2
    assert all(result.title for result in results)
    assert any(step_name == "imputer" for step_name, _ in experiment._effective_numeric_steps())
    assert any(step_name == "power_transformer" for step_name, _ in experiment._effective_numeric_steps())
    assert len(experiment.theory_log) >= 2


def test_experiment_supports_categorical_pipeline_steps():
    df = _classification_df()
    df["crop_type"] = np.where(df["x1"] > 0, "wheat", "corn")

    project = Project(name="irrigation", target_col="target", df=df)
    experiment = Experiment(name="cat-steps", project=project, cv=3)
    experiment.add_categorical_step("onehot", OneHotEncoder(handle_unknown="ignore"))
    experiment.add_model("logreg", LogisticRegression(max_iter=2000), {"C": [0.5, 1.0]})

    experiment.run()
    preds = experiment.predict()

    assert experiment.best_pipeline is not None
    assert any(step_name == "onehot" for step_name, _ in experiment._effective_categorical_steps())
    assert len(preds) > 0


def test_step_addition_replaces_existing_step_name():
    df = _classification_df()
    project = Project(name="demo", target_col="target", df=df)
    experiment = Experiment(name="replace-step", project=project, cv=3)

    experiment.add_numeric_step("imputer", SimpleImputer(strategy="median"))
    experiment.add_numeric_step("imputer", PowerTransformer(method="yeo-johnson"))

    assert len(experiment._effective_numeric_steps()) == 1


def test_project_pipeline_is_default_and_experiment_can_override():
    df = _classification_df()
    project = Project(name="demo", target_col="target", df=df)
    project.add_numeric_step("project_imputer", SimpleImputer(strategy="median"))

    experiment = Experiment(name="default-pipeline", project=project, cv=3)
    default_code = experiment.view_pipeline()
    assert "project_imputer" in default_code

    custom = Experiment(name="custom-pipeline", project=project, cv=3)
    custom.override_pipeline()
    custom.add_numeric_step("experiment_power", PowerTransformer(method="yeo-johnson"))
    custom_code = custom.view_pipeline()
    assert "experiment_power" in custom_code
    assert "project_imputer" not in custom_code


def test_pipeline_is_immutable_after_first_run():
    df = _classification_df()
    project = Project(name="demo", target_col="target", df=df)
    experiment = Experiment(name="lock-test", project=project, cv=3)
    experiment.add_model("logreg", LogisticRegression(max_iter=2000), {"C": [1.0]})

    experiment.run()

    with pytest.raises(RuntimeError):
        experiment.add_numeric_step("late_step", SimpleImputer(strategy="median"))

    with pytest.raises(RuntimeError):
        experiment.use_project_pipeline(False)
