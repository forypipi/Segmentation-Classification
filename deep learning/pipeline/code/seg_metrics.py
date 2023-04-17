import json
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
    model_name = "nnUNet3D"
    fold = "fold_2"
    root = "/data/orfu/DeepLearning/Segmentation-Classification/deep learning/seg/data/output"
    # model_path = os.path.join(model_name, fold)
    model_path = "nnUNet_ensemble"
    path = os.path.join(root, model_path, "summary.json")
    print(path)
    with open(path) as f:
        summary = json.load(f)
        metric = summary["foreground_mean"]
        dice = metric["Dice"]
        iou = metric["IoU"]
        precision = metric["TP"] / (metric["TP"] + metric["FP"])
        recall = metric["TP"] / (metric["TP"] + metric["FN"])
        accuracy = (metric["TP"] + metric["TN"]) / (metric["TP"] + metric["FN"] + metric["FP"] + metric["TN"])
        f1 = 2 * (precision*recall) / (precision + recall)
        print(f"dice: {dice:.5f}")
        print(f"iou: {iou:.5f}")
        print(f"precision: {precision:.5f}")
        print(f"recall: {recall:.5f}")
        print(f"accuracy: {accuracy:.5f}")
        print(f"f1: {f1:.5f}")

# nnUNet2D_0
# dice: 0.74816
# iou: 0.64947
# precision: 0.90908
# recall: 0.80594
# accuracy: 0.99991
# f1: 0.85441

# nnUNet2D_1
# dice: 0.73740
# iou: 0.63633
# precision: 0.90921
# recall: 0.78853
# accuracy: 0.99990
# f1: 0.84458

# nnUNet2D_2
# dice: 0.74502
# iou: 0.64670
# precision: 0.91334
# recall: 0.80692
# accuracy: 0.99990
# f1: 0.85684

# nnUNet3D_0
# dice: 0.73221
# iou: 0.61288
# precision: 0.83738
# recall: 0.78791
# accuracy: 0.99987
# f1: 0.81190

# nnUNet3D_1
# dice: 0.72071
# iou: 0.60203
# precision: 0.84410
# recall: 0.76052
# accuracy: 0.99987
# f1: 0.80014

# nnUNet3D_2
# dice: 0.73327
# iou: 0.61359
# precision: 0.85026
# recall: 0.76872
# accuracy: 0.99987
# f1: 0.80744

# nnUNet ensemble
# dice: 0.76107
# iou: 0.65399
# precision: 0.91370
# recall: 0.77617
# accuracy: 0.99990
# f1: 0.83934