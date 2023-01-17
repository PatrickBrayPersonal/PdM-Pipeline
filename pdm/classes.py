from dataclasses import dataclass


@dataclass
class EvaluationResults:
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
    evaluate: str
    model: str


@dataclass
class PipelineOutput:
    evaluation_results: EvaluationResults
    config: TrainConfig
