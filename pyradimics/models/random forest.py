import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
    'criterion': ["gini", "entropy"],
    # 'n_estimators':range(10, 91, 20),
    'max_depth': range(10, 21, 1),
    'min_samples_split': range(4, 41, 4),
    'min_samples_leaf': range(5, 51, 5),
    'max_features': ["sqrt", "log2", None]
}

# print(sorted(sklearn.metrics.SCORERS.keys()))
grid_search = GridSearchCV(RandomForestClassifier(n_estimators=50, random_state=rand, verbose=1), tuned_parameters, cv=5, n_jobs=8, scoring="accuracy")
grid_search.fit(train_X, train_y)
clf = grid_search.best_estimator_
print(f'best score: {np.round(grid_search.best_score_, 4)}')
print(f'best parameter: {grid_search.best_params_}')
joblib.dump(clf, "../ModelFiles/random forest.m")
metric.FitMetric(clf, test_X, test_y, "rf")

# best score: 0.7577
# accuracy score: 0.7073
# macro precision score: 0.6936
# macro recall score: 0.702
# macro F1 score: 0.6902
# roc_auc score: 0.8746