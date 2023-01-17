import pandas as pd
from sklearn.preprocessing import StandardScaler


def standard_scaler(df: pd.DataFrame) -> pd.DataFrame:
    ss = StandardScaler()
    return ss.fit_transform(df)
