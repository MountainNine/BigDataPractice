from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import time

cancer = load_breast_cancer()
params = {
    'learning_rate': [0.05, 0.1],
    'n_estimators' : [100,500]
}
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=121)
gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train, y_train)
gridCV = GridSearchCV(gb_clf, param_grid=params, cv=2, verbose=1)
gridCV.fit(X_train, y_train)

param = gridCV.best_params_
accuracy =gridCV.best_score_

print(param)
print(accuracy)