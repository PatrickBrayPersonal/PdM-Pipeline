"""
This module defines the following routines used by the 'train' step of the regression recipe:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model recipe.
"""
from typing import Any, Dict

from sklearn.linear_model import LinearRegression


def estimator_fn(estimator_params: Dict[str, Any] = {}):
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    #
    # FIXME::OPTIONAL: return a scikit-learn-compatible regression estimator with fine-tuned
    #                  hyperparameters.

    raise NotImplementedError


def linear_regression(estimator_params: Dict[str, Any] = {}):
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    #
    # FIXME::OPTIONAL: return a scikit-learn-compatible regression estimator with fine-tuned
    #                  hyperparameters.

    return LinearRegression(**estimator_params)
