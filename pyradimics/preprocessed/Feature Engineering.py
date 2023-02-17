import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns


data = pd.read_csv("../data/features.csv", index_col=0)
data = data[data.ct_original_ngtdm_Coarseness != 1000000]    # outlier data
print(data)
x = data.drop(columns="label")
y = data.loc[:, "label"].map({'LUNG_CANCER': 0, 'LYMPHOMA': 1, 'MELANOMA': 2})
sns.countplot(y)
plt.show()

# delete duplicate columns
corrDet=x.corr()
sns.heatmap(corrDet,annot=False)
plt.show()

corrDet.to_csv("../data/correlation.csv")
# print(corrDet)
group = [{i} for i in corrDet.columns]
for x_axis in range(len(corrDet)):
    feature_x = sorted(corrDet.index)[x_axis]
    for feature_y in sorted(corrDet.columns)[x_axis+1: ]:
        val = corrDet.loc[feature_x, feature_y]
        for sets in group:
            if feature_x in sets:
                if feature_y in sets:
                    break
                else:
                    for feature in sets:
                        if corrDet.loc[feature_y, feature] < 0.95:
                            break
                    else:
                        sets.add(feature_y)
delete = set()
cnt = dict()
new_group = group.copy()
while True:
    new_group = [i for i in new_group if len(i)!=1]
    cnt = dict()
    for sets in new_group:
        for ele in sets:
            cnt[ele] = cnt.get(ele, 0) + 1
    for sets in new_group:
        for ele in sets:
            if cnt[ele] != 1:
                delete.add(ele)
                sets.remove(ele)
                break
    for key, val in cnt.items():
        if val > 1:
            break
    else:
        break
for sets in new_group:
    while len(sets) > 1:
        delete.add(sets.pop())
x = x.drop(columns=delete)
print(x)

# select top 100 related columns
b = dict()
for i in x.columns:
    # c,d  = pearsonr(df['Age'],df['Outcome'])
    b[i] = pearsonr(x[i], y)[0]
pearsonrResult = pd.DataFrame(sorted(b.items(), key=lambda x:abs(x[1]), reverse=True), columns=['feature name', 'pearsonr value'])
pearsonrResult.to_csv("../data/pearsonr result.csv", index=False)
selectedFeature = pearsonrResult.loc[:99, "feature name"]
print(selectedFeature)
x = x.loc[:, selectedFeature]
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
# for i, name in enumerate(selectedFeature):
#     if i < 40:
#         continue
#     plt.hist(x = x.loc[:, name], bins = 100)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title(str(i) + " " + name)
#     plt.show()
