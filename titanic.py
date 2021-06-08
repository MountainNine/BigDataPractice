import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import skew
from xgboost import XGBClassifier

df = pd.read_csv('data/titanic/train.csv')
test = pd.read_csv('data/titanic/test.csv')
p_id = test['PassengerId']


# Age, Cabin, Embarked

def pre_procession(x):
    x = x.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)
    x['Age'] = x['Age'].fillna(30)
    x['Fare'] = np.log1p(x['Fare'])
    x['Pclass'] = x['Pclass'].apply(str)
    x = pd.get_dummies(x)
    return x


x = df.drop('Survived', axis=1)
y = df['Survived']

x = pre_procession(x)
test = pre_procession(test)

params = {
    'max_depth': [15, 18, 20, 21, 24],
    'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05]
}

xgb = XGBClassifier(learning_rate=0.02, max_depth=20)

cross_val = cross_val_score(xgb, x, y, scoring='roc_auc', cv=5)
print(np.mean(cross_val))

# grid_cv = GridSearchCV(xgb, param_grid=params, scoring='roc_auc', cv=5)
# grid_cv.fit(x,y)
# print(grid_cv.best_params_)

xgb.fit(x, y)
pred = xgb.predict(test)

# result = pd.DataFrame(p_id)
# result['Survived'] = pred
# result.to_csv('result.csv', index=False)
