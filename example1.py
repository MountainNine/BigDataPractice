import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# MinMaxScaler
def convert_minmax(df):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return df

def convert_log1p(df):
    return np.log1p(df)

# 혼동행렬

def confusion_matrix(real, pred):
    cf = confusion_matrix(real, pred)
    return cf

# PCA 차원축소
def get_pca_ratio(df, n):
    from sklearn.decomposition import PCA
    pca = PCA(n_component=n)
    trans = pca.fit_transform(df)
    return trans.explained_variance_ratio_


# 이상치 검출
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

