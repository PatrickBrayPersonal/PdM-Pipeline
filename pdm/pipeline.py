"""
This is the module in which you define your pipeline functions.

Feel free to break these definitions into as many files as you want for your
preferred code structure.
"""
# Sematic
import dataclasses
from pathlib import Path

import pandas as pd
import sematic
import sklearn.metrics as skmetrics
import yaml
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from pdm.classes import EvaluationResults, PipelineOutput, TrainConfig


@sematic.func(inline=True)
def load_pdm_dataset(train: bool, path: Path = Path("data/raw")) -> pd.DataFrame:
    if train:
        df = pd.read_csv(path / "PdM_train.csv")
    else:
        df = pd.read_csv(path / "PdM_test.csv")
    return df


@sematic.func(inline=False)
def train_model(
    config: TrainConfig,
    train_df: pd.DataFrame,
) -> BaseEstimator:
    model = LinearRegression()
    X, y = split_xy(config, train_df)
    model.fit(X=X, y=y)
    return model


def split_xy(config: TrainConfig, df: pd.DataFrame) -> tuple:
    LABELS = ["id", "RUL", "label1", "label2"]
    X = df.drop(labels=LABELS, axis=1)
    y = df[config.label]
    return X, y


@sematic.func(inline=False)
def evaluate_model(
    config: TrainConfig,
    model: BaseEstimator,
    test_df: pd.DataFrame,
) -> EvaluationResults:
    X, y = split_xy(config, test_df)
    y_hat = model.predict(X)
    results = {}
    for metric in dataclasses.fields(EvaluationResults):
        results[metric.name] = getattr(skmetrics, metric.name)(y, y_hat)
    return EvaluationResults(**results)


@sematic.func(inline=True)
def train_eval(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: TrainConfig
) -> EvaluationResults:
    model = train_model(config=config, train_df=train_df)

    evaluation_results = evaluate_model(config=config, model=model, test_df=test_df)
    return evaluation_results


def read_config(config_name: str) -> TrainConfig:
    config = yaml.full_load(open(f"pdm/config/{config_name}.yaml"))
    return TrainConfig(**config)


@sematic.func(inline=True)
def pipeline(config: str) -> PipelineOutput:
    """
    The root function of the pipeline.
    """
    config = read_config(config)

    train_df = load_pdm_dataset(train=True)

    test_df = load_pdm_dataset(train=False)

    evaluation_results = train_eval(train_df, test_df, config)
    return make_output(
        evaluation_results=evaluation_results,
        config=config,
    )


@sematic.func(inline=True)
def make_output(
    evaluation_results: EvaluationResults,
    config: TrainConfig,
) -> PipelineOutput:
    return PipelineOutput(
        evaluation_results=evaluation_results,
        config=config,
    )
