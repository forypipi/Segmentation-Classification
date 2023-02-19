import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import metric

rand = 3
df = pd.read_csv("../data/preprocessed features.csv", index_col=0)
diction = {0: 'LUNG_CANCER', 1: 'LYMPHOMA', 2: 'MELANOMA'}
train, test = train_test_split(df, test_size=0.2, random_state=3)

# dataset
train_X = train.drop(columns="label")
test_X = test.drop(columns="label")
train_y = train["label"]
test_y = test["label"]

# training
tuned_parameters = {
    'learning_rate': np.arange(0.1, 0.55, 0.05),
    'max_depth': [-1] + list(range(5, 18, 3)),
    'num_leaves': range(10, 34, 4)
}

grid_search = GridSearchCV(LGBMClassifier(n_estimators=50, random_state=3, verbose=1), tuned_parameters, cv=5, n_jobs=8, scoring="accuracy")
grid_search.fit(train_X, train_y)
clf = grid_search.best_estimator_
print(f'best score: {np.round(grid_search.best_score_, 4)}')
print(f'best parameter: {grid_search.best_params_}')
joblib.dump(clf, "../ModelFiles/lgm.m")
metric.FitMetric(clf, test_X, test_y, "lgm")

# best score: 0.7791
# accuracy score: 0.7561
# macro precision score: 0.7628
# macro recall score: 0.756
# macro F1 score: 0.753
# roc_auc score 0.9193