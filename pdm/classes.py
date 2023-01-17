from dataclasses import dataclass

import numpy as np


class Metrics:
    pass


@dataclass
class ClassificationMetrics(Metrics):
    confusion_matrix: np.ndarray
    roc_auc_score: float
    top_k_accuracy_score: float
    f1_score: float


@dataclass
class RegressionMetrics(Metrics):
    r2_score: float
    mean_absolute_error: float
    mean_squared_error: float
    mean_absolute_percentage_error: float
    median_absolute_error: float
    max_error: float
    explained_variance_score: float


@dataclass
class TrainConfig:
    label: str
    cleaning_functions: set
    feature_functions: dict
    evaluate: str
    model: str


@dataclass
class PipelineOutput:
    evaluation_results: Metrics
    config: TrainConfig
