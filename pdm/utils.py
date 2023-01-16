import pandas as pd

from pdm.classes import TrainConfig


def split_xy(config: TrainConfig, df: pd.DataFrame) -> tuple:
    LABELS = ["id", "RUL", "label1", "label2"]
    X = df.drop(labels=LABELS, axis=1)
    y = df[config.label]
    return X, y
