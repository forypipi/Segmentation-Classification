import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import metric

rand = 3
df = pd.read_csv("../data/preprocessed features.csv", index_col=0)
diction = {0: 'LUNG_CANCER', 1: 'LYMPHOMA', 2: 'MELANOMA'}
train, test = train_test_split(df, test_size=0.3, random_state=rand)

# dataset
train_X = train.drop(columns="label")
test_X = test.drop(columns="label")
train_y = train["label"]
test_y = test["label"]

# training
tuned_parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 'auto'],'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
    {'kernel': ['poly'], 'degree': [1, 2, 3, 4], 'C': [1, 10, 100, 1000]}
]
# print(sorted(sklearn.metrics.SCORERS.keys()))
grid_search = GridSearchCV(SVC(probability=True, random_state=rand, verbose=1), tuned_parameters, cv=5, n_jobs=8, scoring="accuracy")
grid_search.fit(train_X, train_y)
clf = grid_search.best_estimator_
print(f'best score: {np.round(grid_search.best_score_, 4)}')
print(f'best parameter: {grid_search.best_params_}')
joblib.dump(clf, "../ModelFiles/svc.m")
metric.FitMetric(clf, test_X, test_y, "svm")

# best score: 0.7544
# accuracy score: 0.7642
# macro precision score: 0.7646
# macro recall score: 0.756
# macro F1 score: 0.7488
# roc_auc score 0.9116