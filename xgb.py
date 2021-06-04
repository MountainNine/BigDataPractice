from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = load_breast_cancer()
x_features = dataset.data
y_label = dataset.target
df = pd.DataFrame(data=x_features, columns=dataset.feature_names)
df['target'] = y_label

x_train, x_test, y_train, y_test = train_test_split(x_features, y_label, test_size=0.2, random_state=124)
xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(x_train, y_train)
pred = xgb_wrapper.predict(x_test)
pred = [1 if x>0.5 else 0 for x in pred]

acc = accuracy_score(y_test, pred)
print(acc)