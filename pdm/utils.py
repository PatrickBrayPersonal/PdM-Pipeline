import pprint
from typing import Dict, Tuple

import pandas as pd
import yaml

from pdm import cleaning, models
from pdm.classes import TrainConfig

pp = pprint.PrettyPrinter(indent=2)


def split_xy(config: TrainConfig, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(config.label, axis=1)
    y = df[config.label]
    return X, y


def read_config(config_name: str) -> TrainConfig:
    config = yaml.full_load(open(f"pdm/config/{config_name}.yaml"))
    validate_config(config)
    return TrainConfig(**config)


def validate_config(config: Dict):
    print("***** CONFIG FILE *****")
    pp.pprint(config)
    print("***********************")
    config_keys = set(config.keys())
    true_keys = set(TrainConfig.__annotations__.keys())
    assert len(config_keys) == len(config.keys()), "CONFIG ERROR: Duplicate Keys"
    assert (
        config_keys - true_keys == set()
    ), f"CONFIG ERROR: {config_keys - true_keys} not in TrainConfig class"
    for clean_fxn in config["cleaning_functions"]:
        assert (
            "name" in clean_fxn.keys()
        ), f"CONFIG ERROR: {clean_fxn} missing mandatory key 'name'"
        assert clean_fxn["name"] in dir(
            cleaning
        ), f"CONFIG ERROR: {clean_fxn['name']} not found in cleaning module"
    for model_fxn in config["model"]:
        assert (
            "name" in model_fxn.keys()
        ), f"CONFIG ERROR: {model_fxn} missing mandatory key 'name'"
        assert model_fxn["name"] in dir(
            models
        ), f"CONFIG ERROR: {model_fxn['name']} not found in models module"
