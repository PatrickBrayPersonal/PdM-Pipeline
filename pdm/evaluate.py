import dataclasses

import pandas as pd
import sematic
import sklearn.metrics as skmetrics
from sklearn.base import BaseEstimator

from pdm.classes import ClassificationMetrics, RegressionMetrics, TrainConfig
from pdm.utils import split_xy


@sematic.func(inline=True)
def regression(
    config: TrainConfig,
    model: BaseEstimator,
    test_df: pd.DataFrame,
) -> RegressionMetrics:
    X, y = split_xy(config, test_df)
    y_hat = model.predict(X)
    results = {}
    for metric in dataclasses.fields(RegressionMetrics):
        results[metric.name] = getattr(skmetrics, metric.name)(y, y_hat)
    return RegressionMetrics(**results)


@sematic.func(inline=True)
def classification(
    config: TrainConfig,
    model: BaseEstimator,
    test_df: pd.DataFrame,
) -> ClassificationMetrics:
    X, y = split_xy(config, test_df)
    y_hat = model.predict(X)
    results = {}
    for metric in dataclasses.fields(ClassificationMetrics):
        results[metric.name] = getattr(skmetrics, metric.name)(y, y_hat)
    return ClassificationMetrics(**results)
