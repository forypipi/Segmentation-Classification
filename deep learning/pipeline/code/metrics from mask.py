import json
import os
import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    auc, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import label_binarize
import torch

if __name__=="__main__":
    model_name = "ResNet3D_VNet3D_inference_01_mask"
    mode = "test"
    root = "/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed"
    diagnose = ["LUNG_CANCER", "LYMPHOMA", "MELANOMA"]
    # diagnose = ["NEGATIVE", "LUNG_CANCER", "LYMPHOMA", "MELANOMA"]

    # diagnose = ["LUNG_CANCER"]
    cnt, diction = 0, {}
    for disease in diagnose:
        for index in os.listdir(os.path.join(root, disease)):
        # for index in ["4_0"]:

            path = os.path.join(root, disease, index)
            try:
                label_txt = open(os.path.join(path, "label.txt"), 'r')
                label = int(label_txt.read())
            except:
                label = 1

            if (mode == "inference" and label == 0) or (mode == "test" and label == -1):     # -1 for test,  other for train
            # if True:
                mask = torch.load(os.path.join(path, "seg_pt", "1.pt"))
                
                # predict_label_txt = open(os.path.join(path, "predict_label.txt"), 'r')
                # predict_label = int(predict_label_txt.read())
                # if predict_label == 0:
                #     predict = torch.zeros((128, 128, 128), device='cpu', dtype=int)
                # else:
                #     predict = torch.load(os.path.join(path, "predict_mask_pt", model_name + ".pt"))

                predict = torch.load(os.path.join(path, "predict_mask_pt", model_name + ".pt"))

                mask = mask.numpy().flatten()
                mask[mask==255] = 1
                predict = predict.cpu().numpy()
                predict = predict.astype(int).flatten() 

                intersect = np.sum(mask * predict)
                dice_union = np.sum(mask) + np.sum(predict)
                dice = (2.0 * intersect) / dice_union
                
                iou_union = np.sum(mask) + np.sum(predict) - intersect
                iou = intersect / iou_union

                if np.isnan(dice) or np.isnan(iou):
                    dice = np.float64(0.)
                    iou = np.float64(0.)

                TN = np.sum(np.logical_and(mask == 0, predict == 0))
                FN = np.sum(np.logical_and(mask == 1, predict == 0))
                TP = np.sum(np.logical_and(mask == 1, predict == 1))
                FP = np.sum(np.logical_and(mask == 0, predict == 1))
                diction[str(cnt)] = {
                    "dice": dice.item(), 
                    "IoU": iou.item(), 
                    "TP": TP.item(), 
                    "FP": FP.item(), 
                    "TN": TN.item(), 
                    "FN": FN.item(), 
                    "disease": disease, 
                    "index": index}
                cnt += 1
                print(cnt)


    total_dice, total_iou, total_TN, total_FN, total_TP, total_FP = 0, 0, 0, 0, 0, 0

    for key, val in diction.items():
        total_dice += val["dice"]
        total_iou += val["IoU"]
        total_TN += val["TN"]
        total_FN += val["FN"]
        total_TP += val["TP"]
        total_FP += val["FP"]

    diction["summary"] = {
        "dice": total_dice/len(diction), 
        "IoU": total_iou/len(diction), 
        "TN": total_TN/len(diction),
        "FN": total_FN/len(diction),
        "TP": total_TP/len(diction),
        "FP": total_FP/len(diction)
        }
    
    root_path = "/data/orfu/DeepLearning/Segmentation-Classification/deep learning/pipeline/data/output/ResNet3D_VNet3D"
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    with open(os.path.join(root_path, mode+' summary.json'), 'w') as f:
        json.dump(diction, f)

    path = os.path.join(root_path, mode+' summary.json')
    with open(path) as f:
        summary = json.load(f)
        metric = summary["summary"]
        dice = metric["dice"]
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

# VNet3D
# train
# dice: 0.79775
# iou: 0.68644
# precision: 0.84139
# recall: 0.79946
# accuracy: 0.99988
# f1: 0.81989
# test
# dice: 0.63841
# iou: 0.51653
# precision: 0.85059
# recall: 0.62336
# accuracy: 0.99983
# f1: 0.71946

# UNet3D
# train
# dice: 0.69885
# iou: 0.56966
# precision: 0.76180
# recall: 0.69780
# accuracy: 0.99982
# f1: 0.72840
# test
# dice: 0.62769
# iou: 0.50565
# precision: 0.75070
# recall: 0.65476
# accuracy: 0.99981
# f1: 0.69946

# ResNet3D_nnUNet3D
# dice: 0.65102
# iou: 0.52491
# precision: 0.81639
# recall: 0.66604
# accuracy: 0.99983
# f1: 0.73359

# ResNet_VNet
# test
# dice: 0.57552
# iou: 0.44887
# precision: 0.82257
# recall: 0.59541
# accuracy: 0.99982
# f1: 0.69079