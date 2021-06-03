from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np


def pre_process(df):
    scaler = RobustScaler()
    df[df.columns[~df.columns.isin(['주구매상품', '주구매지점'])]] = scaler.fit_transform(
        df[df.columns[~df.columns.isin(['주구매상품', '주구매지점'])]])
    df = pd.get_dummies(df, columns=['주구매상품'])
    df = pd.get_dummies(df, columns=['주구매지점'])
    df = df.fillna(0)
    df = df.drop('cust_id', axis=1)

    return df




x = pd.read_csv("data/X_train.csv", encoding='euc-kr')
y = pd.read_csv("data/y_train.csv", encoding='euc-kr')
# x_test = pd.read_csv('data/X_test.csv', encoding='euc_kr')

x = pre_process(x)
y = y.drop('cust_id', axis=1)

# goods = x.loc[:,x.columns[7:49]]
# store = x.loc[:,x.columns[49:]]
# pca = PCA(n_components=20)
# pca.fit(goods)
# x_train = x[:3000]
# x_test = x[3000:]
# y_train = y[:3000]
# y_test = y[3000:]

# learning_rate=0.03, n_estimators=100, max_depth=3, min_samples_leaf=20,min_samples_split=5
params = {
    'min_samples_leaf': [2, 4, 8, 12],
    'min_samples_split': [4, 6, 8]
}
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=134)

classifier = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=8, min_samples_split=8)
# grid = GridSearchCV(classifier, param_grid=params, n_jobs=-1, cv=3)
# grid.fit(x_train, y_train)
# print(grid.best_params_)
# print(grid.best_score_)

classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
acc = accuracy_score(y_test, pred)
roc = roc_auc_score(y_test, pred)
print("acc: {0:} roc: {1:}".format(acc, roc))
#  acc: 0.6814285714285714 roc: 0.6118319191080435
