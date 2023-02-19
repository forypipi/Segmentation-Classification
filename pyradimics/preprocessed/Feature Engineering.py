import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import preprocessing

data = pd.read_csv("../data/features.csv", index_col=0)
data = data[data.ct_original_ngtdm_Coarseness != 1000000]    # outlier data
# delete rows whose ct_original_ngtdm_Coarseness == 1e6
data[data["ct_original_ngtdm_Coarseness"]==1e6] = np.nan
data[data["pet_original_ngtdm_Coarseness"]==1e6] = np.nan
data.dropna(axis=0, how='any', inplace=True)
print(f"delete nan rows: {data.shape}")

x = data.drop(columns="label")
y = data.loc[:, "label"].map({'LUNG_CANCER': 0, 'LYMPHOMA': 1, 'MELANOMA': 2})
print(f"label count: {dict(y.value_counts())}")

# delete duplicate columns
corrDet=x.corr()
# sns.heatmap(corrDet,annot=False)
# plt.show()
corrDet.to_csv("../data/correlation.csv")

corr_threshold = 0.90
label_threshold = 0.10
# group = [{i} for i in corrDet.columns]
delete = set()
for x_axis in range(len(corrDet)):
    feature_x = sorted(corrDet.index)[x_axis]
    for feature_y in sorted(corrDet.columns)[: x_axis]:
        if corrDet.loc[feature_x, feature_y] > corr_threshold:
            delete.add(feature_x)
            break
        # val = corrDet.loc[feature_x, feature_y]
        # for sets in group:
        #     if feature_x in sets:
        #         if feature_y in sets:
        #             break
        #         else:
        #             for feature in sets:
        #                 if corrDet.loc[feature_y, feature] < 0.85:
        #                     break
        #             else:
        #                 sets.add(feature_y)
# delete = set()
# cnt = dict()
# new_group = group.copy()
# while True:
#     new_group = [i for i in new_group if len(i)!=1]
#     cnt = dict()
#     for sets in new_group:
#         for ele in sets:
#             cnt[ele] = cnt.get(ele, 0) + 1
#     for sets in new_group:
#         for ele in sets:
#             if cnt[ele] != 1:
#                 delete.add(ele)
#                 sets.remove(ele)
#                 break
#     for key, val in cnt.items():
#         if val > 1:
#             break
#     else:
#         break
# for sets in new_group:
#     while len(sets) > 1:
#         delete.add(sets.pop())
x = x.drop(columns=delete)
print(f"after delete corr pearson value > {corr_threshold}: {x.shape}")

# select top 100 related columns
b = dict()
for i in x.columns:
    # c,d  = pearsonr(df['Age'],df['Outcome'])
    b[i] = pearsonr(x[i], y)[0]
pearsonResult = pd.DataFrame(sorted(b.items(), key=lambda x:abs(x[1]), reverse=True), columns=['feature name', 'pearson'])
pearsonResult.to_csv("../data/pearson result.csv", index=False)
selectedFeature = pearsonResult[abs(pearsonResult["pearson"])>0.1]
x = x.loc[:, selectedFeature['feature name']]
print(f"after delete label pearson value > {label_threshold}: {x.shape}")

# if "ct_original_ngtdm_Coarseness" in x.columns:
#     x[x["ct_original_ngtdm_Coarseness"] == 1000000] = 0
# for i, name in enumerate(selectedFeature):
#     if i < 40:
#         continue
#     plt.hist(x = x.loc[:, name], bins = 75)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title(str(i) + " " + name)
#     plt.show()

# Normalized
min_max_scaler = preprocessing.MinMaxScaler()
x = pd.DataFrame(min_max_scaler.fit_transform(x), columns=x.columns, index=x.index)
y.index = x.index
x["label"] = y
x.to_csv("../data/preprocessed features.csv")
print(f"preprocessed features: {x.shape}")
# for i, name in enumerate(selectedFeature):
#     if i < 40:
#         continue
#     plt.hist(x = x.loc[:, name], bins = 100)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title(str(i) + " " + name)
#     plt.show()
