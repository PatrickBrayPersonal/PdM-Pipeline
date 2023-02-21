"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
from pandas import DataFrame, Series


def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    #
    # FIXME::OPTIONAL: return a scikit-learn-compatible transformer object.
    #
    # Identity feature transformation is applied when None is returned.
    return None


def drop_columns():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """

    class Dropper:
        def __init__(self):
            pass

        def fit(self, dataset: DataFrame, label: Series):
            self.columns = ["label1", "label2", "id"]

        def transform(self, dataset: DataFrame):
            return dataset.drop(self.columns, axis=1)

    return Dropper()
