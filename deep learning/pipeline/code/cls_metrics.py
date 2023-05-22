import json
import os
import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    auc, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
import torch

def split_train_val_test(clinical: pd.DataFrame, test_radio=0.1, kFold=3):
    train_set, _ = train_test_split(clinical, test_size=test_radio, stratify=clinical['class'], random_state=42)
    train_set.insert(clinical.shape[1], 'training_label', np.nan)
    X, y = train_set['name'], train_set['class']
    skf = StratifiedKFold(n_splits=kFold, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(skf.split(X, y)):
        train_set.iloc[val_idx, 2] = [fold] * len(val_idx)
    train_set = train_set.drop("class", axis=1)
    clinical = clinical.merge(train_set, how='left', left_on='name', right_on='name')
    clinical['training_label'].fillna(-1, inplace=True)
    clinical['training_label'] = clinical['training_label'].astype('int')
    clinical.to_csv("./pipeline/data/training label.csv")
    return clinical

if __name__=="__main__":

    mode = "test"
    root = "/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed"
    diction = {'NEGATIVE': 0, 'LYMPHOMA': 1, 'MELANOMA': 2, 'LUNG_CANCER': 3}
    diagnose = ['NEGATIVE', "LYMPHOMA", "MELANOMA", "LUNG_CANCER"]
    kFold = 5

    dfs = [pd.DataFrame({"class": [label] * len(os.listdir(os.path.join(root, label))), "name": os.listdir(os.path.join(root, label))}) for label in diagnose]
    Whole_df = pd.concat(dfs)
    classes = split_train_val_test(clinical=Whole_df, kFold=kFold, test_radio=0.1)

    target = classes[classes['training_label'].isin([-1]) & classes['class'].isin(diagnose)]

    true, pred = [], []
    for patient in target.values:
        patient_disease, name = patient[0], patient[1]

        true.append(diction[patient_disease])
        predict_label_txt = open(os.path.join(root, patient_disease, name, "predict_label.txt"), 'r')
        pred.append(int(predict_label_txt.read()))

    print(f"{classification_report(true, pred, digits=4)}")
    print(f"accuracy score: {np.round(accuracy_score(true, pred), 4)}")
    print(f"macro precision score: {np.round(precision_score(true, pred, average='macro'), 4)}")
    print(f"macro recall score: {np.round(recall_score(true, pred, average='macro'), 4)}")
    print(f"macro F1 score: {np.round(f1_score(true, pred, average='macro'), 4)}")

# UNet:
# train acc: 0.9726
# accuracy score: 0.549
# macro precision score: 0.4672
# macro recall score: 0.459
# macro F1 score: 0.4627
# roc_auc score: 0.7274

# VNet:
# accuracy score: 0.451
# macro precision score: 0.4013
# macro recall score: 0.3911
# macro F1 score: 0.3912
# roc_auc score: 0.6624

# ResNet3D_PET
# lr_0.0005_WeightDecay_0.01_depth_2.0_PoolSize_4.0_threshold_0.1_epoch_34.0
# train accuracy: 0.9965
# accuracy score: 0.75
# macro precision score: 0.7525
# macro recall score: 0.7601
# macro F1 score: 0.7456
# roc_auc score: 0.9181

# ResNet3D_CT
# lr_0.0005_WeightDecay_0.02_depth_2.0_PoolSize_4.0_threshold_0.1_epoch_40.0
# train accuracy: 0.996
# accuracy score: 0.7063
# macro precision score: 0.7047
# macro recall score: 0.703
# macro F1 score: 0.6991
# roc_auc score: 0.9156

# ResNet3D_PETCT
# lr_0.0005_WeightDecay_0.02_depth_2.0_PoolSize_4.0_threshold_0.15_epoch_60.0
# train accuracy: 0.9868
# accuracy score: 0.7302
# macro precision score: 0.7389
# macro recall score: 0.7334
# macro F1 score: 0.7302
# roc_auc score: 0.9217

# DenseNet3D_CT
# lr_0.0001_WeightDecay_0.01_depth_4.0_PoolSize_4.0_threshold_0.15_epoch_59.0
# training accuracy: 0.9934
# accuracy score: 0.6667
# macro precision score: 0.6564
# macro recall score: 0.6516
# macro F1 score: 0.6522
# roc_auc score: 0.8868

# DenseNet3D_PET
# lr_0.0001_WeightDecay_0.01_depth_4.0_PoolSize_4.0_threshold_0.15_epoch_59.0
# training accuracy: 0.9898
# accuracy score: 0.7937
# macro precision score: 0.7967
# macro recall score: 0.8039
# macro F1 score: 0.7924
# roc_auc score: 0.9346

# DenseNet3D_PETCT
# lr_0.0001_WeightDecay_0.01_depth_4.0_PoolSize_4.0_threshold_0.15_epoch_59.0
# training accuracy: 0.9726
# accuracy score: 0.7024
# macro precision score: 0.7013
# macro recall score: 0.7002
# macro F1 score: 0.6851
# roc_auc score: 0.9197

# ViT3D_CT
# training accuracy: 0.8234
# accuracy score: 0.5198
# macro precision score: 0.5351
# macro recall score: 0.5046
# macro F1 score: 0.4864
# roc_auc score: 0.8116

# ViT3D_PET
# training accuracy: 0.7351
# accuracy score: 0.496
# macro precision score: 0.5068
# macro recall score: 0.5018
# macro F1 score: 0.4935
# roc_auc score: 0.7603

# ViT3D_PETCT
# training accuracy: 0.9682
# accuracy score: 0.5476
# macro precision score: 0.5508
# macro recall score: 0.5399
# macro F1 score: 0.5417
# roc_auc score: 0.8128

# SwinTransformer3D_CT
# training accuracy: 0.9991
# accuracy score: 0.5675
# macro precision score: 0.5775
# macro recall score: 0.5649
# macro F1 score: 0.5659
# roc_auc score: 0.8343

# SwinTransformer3D_PET
# training accuracy: 0.9837
# accuracy score: 0.5754
# macro precision score: 0.5898
# macro recall score: 0.5624
# macro F1 score: 0.5648
# roc_auc score: 0.8469

# SwinTransformer3D_PETCT
# training accuracy: 0.9987
# accuracy score: 0.5635
# macro precision score: 0.5935
# macro recall score: 0.5587
# macro F1 score: 0.5585
# roc_auc score: 0.8501

# pipeline
# accuracy score: 0.6429
# macro precision score: 0.6516
# macro recall score: 0.6389
# macro F1 score: 0.6362
