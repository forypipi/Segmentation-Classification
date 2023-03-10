import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    auc, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import label_binarize


def FitMetric(clf, test_X, test_y, pic_name, pic_dir = "..\pic\pic data"):
    diction = {0: 'LUNG_CANCER', 1: 'LYMPHOMA', 2: 'MELANOMA'}
    prediction = clf.predict(test_X)
    print(f"measure result:{classification_report(test_y, prediction, digits=4)}")
    print(f"accuracy score: {np.round(accuracy_score(test_y, prediction), 4)}")
    print(f"macro precision score: {np.round(precision_score(test_y, prediction, average='macro'), 4)}")
    print(f"macro recall score: {np.round(recall_score(test_y, prediction, average='macro'), 4)}")
    print(f"macro F1 score: {np.round(f1_score(test_y, prediction, average='macro'), 4)}")

    # roc_auc
    Y_pred_prob = clf.predict_proba(test_X)
    test_y = label_binarize(test_y, classes=[0, 1, 2])
    print(f"roc_auc score: {np.round(roc_auc_score(test_y, Y_pred_prob, multi_class='ovr'), 4)}")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):  # 3 classes
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], Y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    color = ['blue', 'grey', 'r']
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), color='black', linestyle='--')
    for i in range(3):
        plt.plot(fpr[i], tpr[i], color=color[i], label=f'label:{diction[i]}, auc={roc_auc[i]:0.4f}')

    # micro
    fpr["micro"], tpr["micro"], thresholds = roc_curve(test_y.ravel(), Y_pred_prob.ravel())
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
    plt.show()

    for key, val in fpr.items():
        key = diction.get(key, key)
        with open(pic_dir+r"\fpr_"+str(key)+".txt", "a") as f:
            f.write(pic_name+": "+str(val.tolist())+"\n")
    for key, val in tpr.items():
        key = diction.get(key, key)
        with open(pic_dir+r"\tpr_"+str(key)+".txt", "a") as t:
            t.write(pic_name+": "+str(val.tolist())+"\n")