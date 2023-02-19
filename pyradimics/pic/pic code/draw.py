import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc

for pic_name in ["LUNG_CANCER", "LYMPHOMA", "MELANOMA", "macro", "micro"]:
    fpr_file = open(os.path.join("..\pic data", "fpr_"+pic_name) + ".txt", "r")
    tpr_file = open(os.path.join("..\pic data", "tpr_"+pic_name) + ".txt", "r")
    fpr_list = fpr_file.readlines()
    tpr_list = tpr_file.readlines()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    color = ['blue', 'grey', 'r', 'g']
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), color='black', linestyle='--')

    for i in range(4):
        ftmp = fpr_list[i][:-1].split(": ")
        fpr[ftmp[0]] = eval(ftmp[1])
        ttmp = tpr_list[i][:-1].split(": ")
        tpr[ttmp[0]] = eval(ttmp[1])
    # print(fpr)
    for i, (key, value) in enumerate(fpr.items()):  # 4 models
        roc_auc[i] = auc(fpr[key], tpr[key])
        print(fpr[key], tpr[key])
        plt.plot(fpr[key], tpr[key], color=color[i], label=f'{key}, auc={roc_auc[i]:0.4f}')

    plt.legend(loc='lower right')
    plt.title(pic_name)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.savefig(os.path.join("..", pic_name) + ".png")
    plt.show()