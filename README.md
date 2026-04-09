# ds-tutor

`ds-tutor` is a lightweight framework for guided scikit-learn experimentation.

## Architecture

The package is organized around two core objects:

- `Project`: immutable dataset context and train/test split, plus default pipeline steps shared by experiments.
- `Experiment`: search plan + results artifact. An experiment can evaluate multiple models and hyperparameter grids, then retain a leaderboard, best pipeline, best params, and scores.

Tutors (e.g. missing data and skewness) inspect the `Project` and declaratively update `Experiment` preprocessing config before execution.

## Quickstart

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from ds_tutor import Project, Experiment, ExploratoryDataAnalysisTutor

df = pd.read_csv("train.csv")
project = Project(name="titanic", target_col="Survived", df=df)
project.add_categorical_step("onehot", OneHotEncoder(handle_unknown="ignore"))

experiment = Experiment(name="baseline-search", project=project, cv=5)
experiment.add_model("logreg", LogisticRegression(max_iter=2000), {
    "C": [0.1, 1.0, 10.0],
})
experiment.add_model("rf", RandomForestClassifier(random_state=42), {
    "max_depth": [None, 8, 16],
    "n_estimators": [100, 200],
})

tutor = ExploratoryDataAnalysisTutor(project, experiment)
tutor.teach_me()
tutor.summary()
experiment.view_pipeline()  # print generated pipeline code
experiment.run()

print(experiment.leaderboard())
print(experiment.best_model_name, experiment.best_params, experiment.best_score)
```

After `experiment.run()` succeeds once, the experiment pipeline becomes immutable.

## Jupyter Summary

```python
tutor.summary()      # EDA Diagnostics + Theory tabs
experiment.summary()
```

`tutor.summary()` renders a clean tabbed EDA view:
- `EDA Diagnostics`: triggered checks, findings, and per-feature detail tables.
- `Theory`: deeper conceptual notes for each check.

`experiment.summary()` renders leaderboard, tutor theory/actions, and the best pipeline diagram.
