import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import skew

df = pd.read_csv('data/titanic/train.csv')
test = pd.read_csv('data/titanic/test.csv')
p_id = test['PassengerId']


# Age, Cabin, Embarked

def pre_procession(x):
    x = x.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)
    x['Age'] = x['Age'].fillna(30)
    x['Fare'] = x['Fare'].fillna(0)
    x['Fare'] = np.log1p(x['Fare'])
    x['Pclass'] = x['Pclass'].apply(str)
    x = pd.get_dummies(x)
    return x


x = df.drop('Survived', axis=1)
y = df['Survived']

x = pre_procession(x)
test = pre_procession(test)

params = {
    'max_depth': [9],
    'max_leaf_nodes': [15],
    'min_samples_leaf': [4],
    'min_samples_split': [9]
}

rfc = RandomForestClassifier(max_depth=9, max_leaf_nodes=15, min_samples_leaf=4, min_samples_split=9, random_state=23)
# cross_val = cross_val_score(rfc, x, y, scoring='roc_auc', cv=5)
# print(np.mean(cross_val))

# grid_cv = GridSearchCV(rfc, param_grid=params, scoring='roc_auc', cv=5, n_jobs=-1)
# grid_cv.fit(x,y)
# print(grid_cv.best_params_, grid_cv.best_score_)

rfc.fit(x, y)
pred = rfc.predict(test)

result = pd.DataFrame(p_id)
result['Survived'] = pred
result.to_csv('result.csv', index=False)
# 0.8553535021725265 0.8664427748471837 0.8670836150756491