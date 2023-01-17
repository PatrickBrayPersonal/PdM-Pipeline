"""
This is the module in which you define your pipeline functions.

Feel free to break these definitions into as many files as you want for your
preferred code structure.
"""
# Sematic
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator

from pdm import evaluate, features, models
from pdm.classes import Metrics, PipelineOutput, TrainConfig
from pdm.utils import read_config, split_xy


def load_pdm_dataset(train: bool, path: Path = Path("data/raw")) -> pd.DataFrame:
    if train:
        df = pd.read_csv(path / "PdM_train.csv")
    else:
        df = pd.read_csv(path / "PdM_test.csv")
    return df


def train_model(
    config: TrainConfig,
    train_df: pd.DataFrame,
) -> BaseEstimator:
    model = getattr(models, config.model)()
    X, y = split_xy(config, train_df)
    model.fit(X=X, y=y)
    return model


def make_features(config: TrainConfig, df: pd.DataFrame) -> pd.DataFrame:
    # for feature in config.feature_functions:
    #     getattr(features, feature)(df)
    return df


def pipeline(config: str) -> PipelineOutput:
    """
    The root function of the pipeline.
    """
    config = read_config(config)

    train_df = load_pdm_dataset(train=True)
    # train_df = make_features(config, train_df)

    test_df = load_pdm_dataset(train=False)
    # test_df = make_features(config, test_df)

    model = train_model(config, train_df)

    evaluation_results = getattr(evaluate, config.evaluate)(
        config=config, model=model, test_df=test_df
    )
    return make_output(
        evaluation_results=evaluation_results,
        config=config,
    )


def make_output(
    evaluation_results: Metrics,
    config: TrainConfig,
) -> PipelineOutput:
    return PipelineOutput(
        evaluation_results=evaluation_results,
        config=config,
    )
