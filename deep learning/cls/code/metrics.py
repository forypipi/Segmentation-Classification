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
    model = "ResNet3D"
    test_df = pd.read_csv(os.path.join("./cls/data/output", model, "test_result.csv"), index_col=0)
    pred = test_df.loc[:, "pred"]
    true = test_df.loc[:, "true"]
    prob = test_df.loc[:, "pred prob"]
    prob = np.array(list(map(eval, prob)))
    FitMetric(pred, true, prob, model)

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

# ResNet3D
# training acc: 1
# accuracy score: 0.9083
# macro precision score: 0.8955
# macro recall score: 0.8388
# macro F1 score: 0.8591
# roc_auc score: 0.98