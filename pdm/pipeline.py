"""
This is the module in which you define your pipeline functions.

Feel free to break these definitions into as many files as you want for your
preferred code structure.
"""
# Sematic
from dataclasses import dataclass
import sematic
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator


@dataclass
class PipelineOutput:
    len_train_df: int
    len_test_df: int
    author: str = "Patrick Bray"


@dataclass
class TrainConfig:
    learning_rate: float = 1
    epochs: int = 14


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
    print("dev")


@sematic.func(inline=True)
def train_eval(
    train_df: pd.DataFrame, test_df: pd.DataFrame, train_config: TrainConfig
):
    model = train_model(config=train_config, train_df=train_df)

    evaluation_results = evaluate_model(
        model=model, test_loader=test_dataloader, device=device
    )
    return evaluation_results


@sematic.func(inline=True)
def pipeline() -> PipelineOutput:
    """
    The root function of the pipeline.
    """
    train_df = load_pdm_dataset(train=True)

    test_df = load_pdm_dataset(train=False)
    return make_output(train_df=train_df, test_df=test_df)


@sematic.func(inline=True)
def make_output(train_df: pd.DataFrame, test_df: pd.DataFrame) -> PipelineOutput:
    return PipelineOutput(len_train_df=len(train_df), len_test_df=len(test_df))
