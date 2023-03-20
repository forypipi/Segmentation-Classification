import os
import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    auc, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import label_binarize


def FitMetric(pred, true, prob, pic_name, result_path="./cls/data/output", pic_dir="./cls/pic/pic data"):
    
    diction = {0: 'NEGATIVE', 1: 'LYMPHOMA', 2: 'MELANOMA', 3: 'LUNG_CANCER'}

    print(f"measure result:{classification_report(true, pred, digits=4)}")
    print(f"accuracy score: {np.round(accuracy_score(true, pred), 4)}")
    print(f"macro precision score: {np.round(precision_score(true, pred, average='macro'), 4)}")
    print(f"macro recall score: {np.round(recall_score(true, pred, average='macro'), 4)}")
    print(f"macro F1 score: {np.round(f1_score(true, pred, average='macro'), 4)}")

    # roc_auc
    true = label_binarize(true, classes=[0, 1, 2, 3])
    print(f"roc_auc score: {np.round(roc_auc_score(true, prob, multi_class='ovr'), 4)}")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):  # 4 classes
        fpr[i], tpr[i], _ = roc_curve(true[:, i], prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    color = ['blue', 'grey', 'r', 'violet']
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), color='black', linestyle='--')
    for i in range(4):
        plt.plot(fpr[i], tpr[i], color=color[i], label=f'label:{diction[i]}, auc={roc_auc[i]:0.4f}')

    # micro
    fpr["micro"], tpr["micro"], thresholds = roc_curve(true.ravel(), prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], color='g', label=f'micro, auc={roc_auc["micro"]:0.4f}')

    # macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= 3
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"], color='yellow', label=f'macro, auc={roc_auc["macro"]:0.4f}')

    plt.legend(loc='lower right')
    plt.title(pic_name)
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.savefig(os.path.join(".", "cls", "pic", pic_name) + ".png")
    plt.show()

    for key, val in fpr.items():
        key = diction.get(key, key)
        with open(os.path.join(pic_dir, r"fpr_"+str(key)+".txt"), "a") as f:
            f.write(pic_name+": "+str(val.tolist())+"\n")
    for key, val in tpr.items():
        key = diction.get(key, key)
        with open(os.path.join(pic_dir, r"tpr_"+str(key)+".txt"), "a") as t:
            t.write(pic_name+": "+str(val.tolist())+"\n")

if __name__=="__main__":
    model_name = "ViT3D_CT"

    grid_search_path = os.path.join("./cls/Performance", model_name)
    validation_df = pd.read_csv(os.path.join(grid_search_path, "grid search.csv"), index_col=0)
    params = validation_df.loc[validation_df['accuracy'].idxmax()].to_dict()
    params.pop('accuracy')
    param_path = "_".join([str(item) for key_value in params.items() for item in key_value])
    print(param_path)

    test_df = pd.read_csv(os.path.join("./cls/data/output", model_name, param_path, "best", "train_result.csv"), index_col=0)
    pred = test_df.loc[:, "pred"]
    true = test_df.loc[:, "true"]
    prob = test_df.loc[:, "pred prob"]
    prob = np.array(list(map(eval, prob)))
    FitMetric(pred, true, prob, model_name)

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
# training accuracy: 0.7351
# accuracy score: 0.496
# macro precision score: 0.5068
# macro recall score: 0.5018
# macro F1 score: 0.4935
# roc_auc score: 0.7603

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