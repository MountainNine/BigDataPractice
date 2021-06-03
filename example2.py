from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np


def get_outlier(df, columns, weight=1.5):
    for column in columns:
        col = df[column]
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)

        iqr = q3 - q1
        iqr_weight = iqr * weight
        low = q1 - iqr_weight
        high = q3 + iqr_weight
        df.loc[df[column] < low, column] = low
        df.loc[df[column] > high, column] = high
    return df

def pre_processing(x):
    x = x.fillna(0)
    scaler = StandardScaler()
    x[['총구매액', '최대구매액', '환불금액']] = np.log(x[['총구매액', '최대구매액', '환불금액']] + 1)
    x = get_outlier(x, ['총구매액', '최대구매액', '환불금액'])

    x[x.columns[~x.columns.isin(['주구매상품', '주구매지점'])]] = scaler.fit_transform(
        x[x.columns[~x.columns.isin(['주구매상품', '주구매지점'])]])

    # etc_goods = x['주구매상품'].value_counts()[x['주구매상품'].value_counts() < 50].keys()
    # etc_store = x['주구매지점'].value_counts()[x['주구매지점'].value_counts() < 50].keys()
    # x.loc[x['주구매상품'].isin(etc_goods),'주구매상품'] = '그외상품'
    # x.loc[x['주구매지점'].isin(etc_store),'주구매지점'] = '그외지점'
    x = pd.get_dummies(x, columns=['주구매상품'])
    x = pd.get_dummies(x, columns=['주구매지점'])
    if True in x.columns.isin(['주구매상품_소형가전']):
        x = x.drop('주구매상품_소형가전', axis=1)
    x = x.drop(['cust_id'], axis=1)
    return x


x_origin = pd.read_csv("data/X_train.csv", encoding='euc-kr')
y = pd.read_csv("data/y_train.csv", encoding='euc-kr')
# x_test = pd.read_csv('data/X_test.csv', encoding='euc_kr')

# 'cust_id', '총구매액', '최대구매액', '환불금액', '주구매상품', '주구매지점', '내점일수', '내점당구매건수',
# '주말방문비율', '구매주기'

x = pre_processing(x_origin)
# x_test = pre_processing(x_test)
y = y.drop('cust_id', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=123)
params = {
    'learning_rate': [0.05]
}
classifier = XGBClassifier(learning_rate=0.03, max_depth=4, n_estimators=100)
# grid = GridSearchCV(classifier, param_grid=params, n_jobs=-1, cv=3)
# grid.fit(x_train, y_train)
# print(grid.best_params_)
# print(grid.best_score_)

classifier.fit(x_train, y_train)
pred = classifier.predict_proba(x_test)
result = pd.DataFrame(pred[:,1])

pred = classifier.predict(x_test)
acc = accuracy_score(y_test, pred)
roc = roc_auc_score(y_test, pred)
print("acc: {0:} roc: {1:}".format(acc, roc))
 # acc: 0.6614285714285715 roc: 0.623664703185683
