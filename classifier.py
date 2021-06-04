from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    cols = ['총구매액', '최대구매액', '환불금액']
    x[cols] = np.log(x[cols] + 1)
    x = get_outlier(x, cols)

    # x.loc[x['구매주기'] == 0,'구매주기'] = 13
    x[x.columns[~x.columns.isin(['주구매상품', '주구매지점'])]] = scaler.fit_transform(
        x[x.columns[~x.columns.isin(['주구매상품', '주구매지점'])]])

    x = pd.get_dummies(x, columns=['주구매상품'])
    x = pd.get_dummies(x, columns=['주구매지점'])
    if True in x.columns.isin(['주구매상품_소형가전']):
        x = x.drop('주구매상품_소형가전', axis=1)
    x = x.drop(['cust_id'], axis=1)
    return x


x_origin = pd.read_csv("data/X_train.csv", encoding='euc-kr')
y = pd.read_csv("data/y_train.csv", encoding='euc-kr')

x = pre_processing(x_origin)
y = y.drop('cust_id', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=134)

classifier = XGBClassifier(learning_rate=0.01, n_jobs=-1, )
classifier.fit(x_train, y_train)
pred = classifier.predict_proba(x_test)
result = pd.DataFrame(pred[:, 1])
print(result)

pred = classifier.predict(x_test)
acc = accuracy_score(y_test, pred)
roc = roc_auc_score(y_test, pred)
print("acc: {0:} roc: {1:}".format(acc, roc))

# acc: 0.6614285714285715 roc: 0.623664703185683
