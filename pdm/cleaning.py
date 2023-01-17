import pandas as pd


def drop_columns(df, cols):
    return df.drop(cols, axis=1)
