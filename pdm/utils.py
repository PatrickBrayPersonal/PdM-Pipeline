import pandas as pd
import yaml

from pdm.classes import TrainConfig


def split_xy(config: TrainConfig, df: pd.DataFrame) -> tuple:
    LABELS = ["id", "RUL", "label1", "label2"]
    X = df.drop(labels=LABELS, axis=1)
    y = df[config.label]
    return X, y


def read_config(config_name: str) -> TrainConfig:
    config = yaml.full_load(open(f"pdm/config/{config_name}.yaml"))
    return TrainConfig(**config)
