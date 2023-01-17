"""
This is the module in which you define your pipeline functions.

Feel free to break these definitions into as many files as you want for your
preferred code structure.
"""
# Sematic
from pathlib import Path

import pandas as pd
import sematic
import yaml
from sklearn.base import BaseEstimator

import pdm
from pdm import evaluate as pdm_evals
from pdm import models
from pdm.classes import EvaluationResults, PipelineOutput, TrainConfig
from pdm.utils import split_xy


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
    model = getattr(models, config.model)()()
    X, y = split_xy(config, train_df)
    model.fit(X=X, y=y)
    return model


@sematic.func(inline=True)
def train_eval(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: TrainConfig
) -> EvaluationResults:
    model = train_model(config=config, train_df=train_df)

    evaluation_results = getattr(pdm.evaluate, config.evaluate)(
        config=config, model=model, test_df=test_df
    )
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
