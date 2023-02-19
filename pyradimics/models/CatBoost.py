import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import metric

rand = 3
df = pd.read_csv("../data/preprocessed features.csv", index_col=0)
diction = {0: 'LUNG_CANCER', 1: 'LYMPHOMA', 2: 'MELANOMA'}
train, test = train_test_split(df, test_size=0.2, random_state=rand)

# dataset
train_X = train.drop(columns="label")
test_X = test.drop(columns="label")
train_y = train["label"]
test_y = test["label"]

# training
tuned_parameters = {
    'depth': [6],
    'learning_rate': [0.03],
    'l2_leaf_reg': [3],
    'iterations': [1000]
}

grid_search = GridSearchCV(CatBoostClassifier(random_state=rand, verbose=True), tuned_parameters, cv=5, n_jobs=8, scoring="accuracy")
grid_search.fit(train_X, train_y)
clf = grid_search.best_estimator_
print(f'best score: {np.round(grid_search.best_score_, 4)}')
print(f'best parameter: {grid_search.best_params_}')
joblib.dump(clf, "../ModelFiles/CatBoost.m")
metric.FitMetric(clf, test_X, test_y, "CatBoost")

# best score: 0.7544
# best parameter: {'depth': 6, 'iterations': 1000, 'l2_leaf_reg': 3, 'learning_rate': 0.03}
# accuracy score: 0.8171
# macro precision score: 0.818
# macro recall score: 0.8159
# macro F1 score: 0.8131
# roc_auc score: 0.9235