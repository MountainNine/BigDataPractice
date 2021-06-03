import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_outlier(df, columns, weight=1.5):
    for column in columns:
        col = df[column]
        q1 = np.quantile(col.values, 0.25)
        q3 = np.quantile(col.values, 0.75)

        iqr = q3 - q1
        iqr_weight = iqr * weight
        low = q1 - iqr_weight
        high = q3 + iqr_weight
        df.loc[df[column] < low, column] = low
        df.loc[df[column] > high, column] = high
    return df

