import os

import kagglehub
import pandas as pd


def kaggle_dataset(url):
    dataset_path = kagglehub.competition_download(url)
    train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(dataset_path, "test.csv"))
    return train_df, test_df
