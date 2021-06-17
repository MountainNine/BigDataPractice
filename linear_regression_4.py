import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


train_x = pd.read_csv('data/mosquito/train_x.csv')
train_y = pd.read_csv('data/mosquito/train_y.csv')
test_x = pd.read_csv('data/mosquito/test_x.csv')

def pre_processing(x):
    x['date'] = list(map(lambda i: pd.to_datetime(i), x['date']))
    x['month'] = list(map(lambda i: i.month, x['date']))

    x['mosquito_term'] = list(map(lambda i: 'Yes' if (4 <= i <= 10) else 'No', x['month']))
    x = pd.get_dummies(x)

    x = x.drop(['date', 'month'], axis=1)
    return x

