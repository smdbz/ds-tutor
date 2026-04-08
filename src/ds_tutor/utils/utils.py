import os

import pandas as pd


def kaggle_dataset(competition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download Kaggle competition data and return (train_df, test_df)."""

    import kagglehub

    dataset_path = kagglehub.competition_download(competition)
    train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(dataset_path, "test.csv"))
    return train_df, test_df
