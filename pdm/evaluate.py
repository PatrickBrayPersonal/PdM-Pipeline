import dataclasses

import pandas as pd
import sematic
import sklearn.metrics as skmetrics
from sklearn.base import BaseEstimator

from pdm.classes import EvaluationResults, TrainConfig
from pdm.utils import split_xy


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
