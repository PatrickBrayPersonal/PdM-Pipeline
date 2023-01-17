"""
This is the module in which you define your pipeline functions.

Feel free to break these definitions into as many files as you want for your
preferred code structure.
"""
# Sematic
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from pdm import cleaning, evaluate, features, models
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
    model: BaseEstimator,
    train_df: pd.DataFrame,
) -> BaseEstimator:
    X, y = split_xy(config, train_df)
    model.fit(X=X, y=y)
    return model


def clean_dataset(config: TrainConfig, df: pd.DataFrame) -> pd.DataFrame:
    for cleaning_func in config.cleaning_functions:
        if "args" in cleaning_func:
            df = getattr(cleaning, cleaning_func["name"])(df, **cleaning_func["args"])
        else:
            df = getattr(cleaning, cleaning_func["name"])(df)
    return df


def make_model(config: TrainConfig) -> BaseEstimator:
    model_list = []
    for model_info in config.model:
        model = getattr(models, model_info["name"])
        if "args" in model_info:
            model = (model_info["name"], model(**model_info["args"]))
        else:
            model = (model_info["name"], model())
        model_list.append(model)
    return Pipeline(model_list)


def pipeline(config: str) -> PipelineOutput:
    """
    The root function of the pipeline.
    """
    config = read_config(config)

    train_df = load_pdm_dataset(train=True)
    train_df = clean_dataset(config, train_df)

    test_df = load_pdm_dataset(train=False)
    test_df = clean_dataset(config, test_df)

    model = make_model(config)
    model = train_model(config, model, train_df)

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
