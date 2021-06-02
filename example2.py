from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from xgboost import XGBClassifier
import numpy as np


def pre_process(df):
    label = LabelEncoder()
    scaler = StandardScaler()
    df['주구매상품'] = label.fit_transform(df['주구매상품'])
    df['주구매지점'] = label.fit_transform(df['주구매지점'])
    df[:] = scaler.fit_transform(df)
    df = df.fillna(0)
    df =df.drop('cust_id', axis=1)
    return df


x = pd.read_csv("data/X_train.csv", encoding='euc-kr')
y = pd.read_csv("data/y_train.csv")
# x_test = pd.read_csv('data/X_test.csv', encoding='euc_kr')

x = pre_process(x)
y = y.drop('cust_id', axis=1)
# x_train = x[:3000]
# x_test = x[3000:]
# y_train = y[:3000]
# y_test = y[3000:]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=134)

classifier = XGBClassifier()
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
acc = accuracy_score(y_test, pred)
print(acc)