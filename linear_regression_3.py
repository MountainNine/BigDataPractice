import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

train_x = pd.read_csv('data/jeju-traffic/train_x.csv')
train_y = pd.read_csv('data/jeju-traffic/train_y.csv')
test_x = pd.read_csv('data/jeju-traffic/test_x.csv')


def preprocessing(x):
    x = x.drop('id', axis=1)
    drop_column = ['시도명', '일자']
    log_cols = ['거주인구', '방문인구', '총 유동인구', '근무인구', '평균 속도', '일강수량', '평균 풍속']
    scale_cols = ['거주인구', '방문인구', '총 유동인구', '근무인구', '평균 속도', '일강수량', '평균 풍속', '평균 소요 시간', '평균 기온']
    x['일자'] = x['일자'].apply(pd.to_datetime)
    x['year'] = x['일자'].apply(lambda i: i.year)
    x['month'] = x['일자'].apply(lambda i: i.month)
    x['day'] = x['일자'].apply(lambda i: i.day)
    for i in log_cols:
        x[i] = np.log1p(x[i])
    scaler = StandardScaler()
    x[scale_cols] = scaler.fit_transform(x[scale_cols])
    x = x.drop(drop_column, axis=1)
    x = pd.get_dummies(x, columns=['읍면동명', 'year', 'month', 'day'])
    return x


x = preprocessing(train_x)
test_id = test_x['id']
test = preprocessing(test_x)
y = train_y.drop('id', axis=1)
ridge_param = {'alpha': [13, 14, 15, 16, 17]}
lasso_param = {'alpha': [0.5, 0.525, 0.55, 0.575, 0.6]}
param = {'min_samples_leaf': [2,4,6,8],
             'max_depth': [3, 6, 12, 15, 20]}
ridge = Ridge(alpha=14)
lasso = Lasso(alpha=0.6)
rfr = RandomForestRegressor(random_state=24)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
rfr.fit(x_train,y_train)
pred = rfr.predict(x_test)
mse = mean_squared_error(y_test, pred)
print(mse)

# rfr.fit(x, y)
# score = GridSearchCV(rfr, param_grid=param, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
# score.fit(x,y)
# print(score.best_params_)
# for model in models:
#     score = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')
#     print(model.__class__.__name__, -1 * np.mean(score))

# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)