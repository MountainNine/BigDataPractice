import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def preprocessing(x):
    cols = ['temp', 'atemp', 'humidity', 'windspeed']
    scaler = StandardScaler()
    x['datetime'] = x.datetime.apply(pd.to_datetime)
    x['year'] = x.datetime.apply(lambda x: x.year)
    x['month'] = x.datetime.apply(lambda x: x.month)
    x['day'] = x.datetime.apply(lambda x: x.day)
    x['hour'] = x.datetime.apply(lambda x: x.hour)
    x = pd.get_dummies(x, columns=['season', 'holiday', 'workingday', 'weather', 'year', 'month', 'hour'])


    x[cols] = scaler.fit_transform(x[cols])
    y = np.log1p(x['count'])
    x = x.drop(['datetime', 'count', 'casual', 'registered'], axis=1)
    return x, y


train = pd.read_csv('data/bike-sharing-demand/train.csv')
x, y = preprocessing(train)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
coef = pd.Series(lr.coef_, index=x_train.columns)

mse = mean_absolute_error(np.expm1(y_test), np.expm1(pred))
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred)))
r2_score = r2_score(np.expm1(y_test), np.expm1(pred))
print(mse, rmse)
print(r2_score)

