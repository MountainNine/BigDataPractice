import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import sklearn
import numpy as np
import warnings

warnings.filterwarnings('ignore')

train_x = pd.read_csv('data/mosquito/train_x.csv')
train_y = pd.read_csv('data/mosquito/train_y.csv')
test_x = pd.read_csv('data/mosquito/test_x.csv')

train_x['date'] = list(map(lambda i: pd.to_datetime(i), train_x['date']))
train_x['month'] = list(map(lambda i: i.month, train_x['date']))
# train_x['mosquito_term'] = list(map(lambda i: 'Yes' if (6 <= i <= 9) else 'No', train_x['month']))
# train_x = pd.get_dummies(train_x)
train_x = train_x.drop(['date'], axis=1)
scaler = StandardScaler()
train_x[['강수량(mm)', '평균기온(℃)', '최저기온(℃)', '최고기온(℃)']] = scaler.fit_transform(
    train_x[['강수량(mm)', '평균기온(℃)', '최저기온(℃)', '최고기온(℃)']])

train_y = train_y.drop('date', axis=1)

rf = RandomForestRegressor(max_depth=4, max_leaf_nodes=16, min_samples_leaf=2, min_samples_split=25, random_state=23)
print(rf.get_params())
cv_score = cross_val_score(rf, train_x, train_y, cv=5, scoring='r2')
print(np.mean(cv_score))

params = {
    'max_depth':[4],
    'max_leaf_nodes': [16],
    'min_samples_leaf':[2],
    'min_samples_split':[25]

}
grid_cv = GridSearchCV(rf, cv=5, param_grid=params, scoring='r2')
grid_cv.fit(train_x, train_y)

print(grid_cv.best_params_, grid_cv.best_score_)
# 0.6798546739969036