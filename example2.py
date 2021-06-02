from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

df = pd.read_csv("data/X_test.csv", encoding='euc-kr')
label = LabelEncoder()
minmax = MinMaxScaler()
df['주구매상품'] = label.fit_transform(df['주구매상품'])
df['주구매지점'] = label.fit_transform(df['주구매지점'])
df[:] = minmax.fit_transform(df)
df = df.fillna(0)
print(df.head())
