import dataclasses

import pandas as pd
import sklearn.metrics as skmetrics
from sklearn.base import BaseEstimator

from pdm.classes import ClassificationMetrics, RegressionMetrics, TrainConfig
from pdm.utils import split_xy


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


def classification(
    config: TrainConfig,
    model: BaseEstimator,
    test_df: pd.DataFrame,
) -> ClassificationMetrics:
    X, y = split_xy(config, test_df)
    y_hat = model.predict(X)
    assert (
        y_hat.min() >= 0 and y_hat.max() <= 1
    ), f"Improper classification result: ranges {y_hat.min()} to {y_hat.max()}"
    assert (
        y.min() >= 0 and y.max() <= 1
    ), f"Improper classification label: ranges {y.min()} to {y.max()}"
    results = {}
    for metric in dataclasses.fields(ClassificationMetrics):
        results[metric.name] = getattr(skmetrics, metric.name)(y, y_hat)
    return ClassificationMetrics(**results)
