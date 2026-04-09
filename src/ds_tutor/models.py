from dataclasses import dataclass
from typing import Any
from sklearn.base import BaseEstimator

@dataclass(frozen=True)
class ModelSpec:
    """
    Configuration for a machine learning model to be evaluated in an Experiment.
    
    Attributes:
        name (str): A descriptive identifier for the model (e.g., 'RandomForest', 'XGBoost').
        estimator (BaseEstimator): An uninitialized scikit-learn compatible estimator.
        param_grid (dict[str, list[Any]]): A dictionary of hyperparameters to search over.
    """
    name: str
    estimator: BaseEstimator
    param_grid: dict[str, list[Any]]
