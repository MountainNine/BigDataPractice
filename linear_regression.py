import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def preprocessing(x):
    cols = ['temp', 'atemp', 'humidity', 'windspeed']
    x = pd.get_dummies(x, columns=['season', 'holiday', 'workingday', 'weather'])
    scaler = StandardScaler()
    x[cols] = scaler.fit_transform(x[cols])
    y = x['count']
    x = x.drop(['datetime', 'count', 'casual', 'registered'], axis=1)
    return x, y


train = pd.read_csv('data/bike-sharing-demand/train.csv')
x, y = preprocessing(train)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=134)
lr = LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print(pred)
print(y_test.values)

mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2_score = r2_score(y_test, pred)
print(mse, rmse)
print(r2_score)