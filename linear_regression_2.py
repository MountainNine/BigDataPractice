import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from scipy.stats import skew

warnings.filterwarnings('ignore')
import numpy as np

df_origin = pd.read_csv('data/house-prices/train.csv')
df = df_origin.copy()

df['SalePrice'] = np.log(df['SalePrice'])

df = df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
df.fillna(df.mean(), inplace=True)
isnull_series = df.isnull().sum()

features_index = df.dtypes[df.dtypes != 'object'].index
skew_feature = df[features_index].apply(lambda x: skew(x))
skew_feature_top = skew_feature[skew_feature > 1]
df[skew_feature_top.index] = np.log1p(df[skew_feature_top.index])

df = pd.get_dummies(df)


def get_rmse(model):
    pred = model.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print(model.__class__.__name__, ': ', np.round(rmse, 3))
    return rmse


def get_rmses(models):
    rmses = []
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses


def print_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=5)
    grid_model.fit(x, y)
    rmse = np.sqrt(-1 * grid_model.best_score_)
    print(model.__class__.__name__, ': ', rmse, ',', grid_model.best_params_)


def print_coef(model, n=10):
    coef = pd.Series(model.coef_, index=x.columns)
    print(coef.sort_values(ascending=False))


ridge_params = {'alpha': [0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]}
lasso_params = {'alpha': [0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1, 5, 10]}

y = df['SalePrice']
x = df.drop(['SalePrice'], axis=1, inplace=False)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=156)

lr = LinearRegression()
lr.fit(x_train, y_train)
ridge = Ridge(alpha=10)
ridge.fit(x_train, y_train)
lasso = Lasso(alpha=0.001)
lasso.fit(x_train, y_train)

models = [lr, ridge, lasso]

# print_best_params(ridge, ridge_params)
# print_best_params(lasso, lasso_params)
print_coef(ridge)
