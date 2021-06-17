import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

train_x = pd.read_csv('data/mosquito/train_x.csv')
train_y = pd.read_csv('data/mosquito/train_y.csv')
test_x = pd.read_csv('data/mosquito/test_x.csv')
print(dir(pd.DataFrame))


def pre_processing(x):
    x['date'] = list(map(lambda i: pd.to_datetime(i), x['date']))
    x['month'] = list(map(lambda i: i.month, x['date']))

    x['mosquito_term'] = list(map(lambda i: 'Yes' if (4 <= i <= 10) else 'No', x['month']))
    x = pd.get_dummies(x)

    x = x.drop(['date', 'month'], axis=1)
    scaler = StandardScaler()
    x[['강수량(mm)','평균기온(℃)','최저기온(℃)','최고기온(℃)']] = scaler.fit_transform(x[['강수량(mm)','평균기온(℃)','최저기온(℃)','최고기온(℃)']])
    return x

train_x = pre_processing(train_x)
train_y = train_y.drop('date', axis=1)

xgb = XGBRegressor(learning_rate=0.025, max_depth=2)
cv_score = cross_val_score(xgb, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
cv_score_mean = np.sqrt(np.mean(cv_score) * -1)
print(cv_score_mean)

# params = {
#     'max_depth': [2,3]
# }
# grid_cv = GridSearchCV(xgb, cv=5, param_grid=params, scoring='neg_mean_squared_error')
# grid_cv.fit(train_x, train_y)
#
# print(grid_cv.best_params_)
