import time

import numpy as np
import torch
from torch import nn, Tensor, optim
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import pandas as pd
import os

from tqdm import tqdm

import loss
from models import UNet3D, VNet

def ReadDcm(path):
    """
    :param path: folder path
    :return: image object, reader object(read metadata)
    """
    reader = sitk.ImageSeriesReader()
    # dicom_names = reader.GetGDCMSeriesFileNames(path)
    # print(dicom_names)
    reader.SetFileNames(path)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    return sitk.GetArrayFromImage(image), reader

def normalized(img, range1=(0, 1)):
    img = img.astype(np.float32)      # int32 to float32
    if img.max() != img.min():
        img = (img - img.min()) / (img.max() - img.min())    # to 0--1
    img = img * (range1[1] - range1[0]) + range1[0]
    return Tensor(img)

def training(model, optimizer, seg_loss, cls_loss, train_loader: DataLoader, verbose=False, EPOCHS=10):
    # log_steps = 1
    # global_step = 0
    # # start = time.time()
    loss_list = []
    running_loss = 0.0
    loop = tqdm(enumerate(train_loader), total =len(train_loader))

    for epoch in range(EPOCHS):
        for step, data in loop:
            images, seg_images, ages, sexs, labels = data
            sexs = sexs.unsqueeze(1)
            sexs = torch.zeros(len(sexs), 3).scatter_(1, sexs, 1)
            # image: [从头向下, 从前向后, 从左向右]
            optimizer.zero_grad()
            segmentation, classification = model(images, sexs, ages)

            # 损失函数采用nn.CrossEntropyLoss()
            # if labels == 0:
            #     train_loss = cls_loss(classification, labels)
            # else:
            train_loss = seg_loss(segmentation, seg_images) + cls_loss(classification, labels)
            running_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{epoch}/{EPOCHS}]')
            loop.set_postfix(loss=running_loss/(step+1))

            # 每100个batch检查一次交叉熵的值
            # global_step += 1
            # if global_step % log_steps == 0:
            #     print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f s/step"
            #           % (global_step, epoch, step, float(train_loss), (time.time() - start) / global_step))
            loss_list.append(float(train_loss))
    if verbose:
        return model, train_loss, pd.DataFrame(loss_list)
    else:
        return model, train_loss


def val(model, val_loader: DataLoader, seg_loss, cls_loss):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for step, data in enumerate(val_loader):
            images, seg_images, ages, sexs, labels = data
            sexs = sexs.unsqueeze(1)
            sexs = torch.zeros(len(sexs), 3).scatter_(1, sexs, 1)
            segmentation, classification = model(images, sexs, ages)
            loss = seg_loss(segmentation, seg_images) + cls_loss(classification, labels)
            total_loss += loss
        avg_loss = total_loss / len(val_loader)
    return avg_loss

class MyData(Dataset):
    def __init__(self, clinical: pd.DataFrame, data_label: pd.DataFrame, label_list: list, root_dir=r'E:\dataset\preprocessed'):
        self.clinical = clinical
        self.data_label = data_label
        self.study_list = self.data_label[self.data_label["training_label"].isin(label_list)].reset_index().loc[:, 'Study UID']
        self.root_dir = root_dir

    def __getitem__(self, item):
        sex_dict = {'F': 0, 'M': 1}
        label_dict = {'NEGATIVE': 0, 'LYMPHOMA': 1, 'MELANOMA': 2, 'LUNG_CANCER': 3}
        study_id = self.study_list[item]
        patient_info = self.clinical[self.clinical['Study UID']==study_id]
        patient_dir = os.path.join(self.root_dir, str(patient_info.index[1] // 3))
        ct_path = os.path.join(patient_dir, 'CT_nii', '1.nii')
        pet_path = os.path.join(patient_dir, 'PET_nii', '1.nii')
        seg_path = os.path.join(patient_dir, 'seg_nii', '1.nii')
        ct_image = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
        pet_image = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))
        seg_image = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        clinical = pd.read_csv(os.path.join(patient_dir, "clinical.csv"))
        age = int(clinical.loc[0, "age"])
        sex = clinical.loc[0, "sex"]
        if isinstance(sex, str):
            sex = sex_dict[sex]
        else:
            sex = 2
        label = label_dict[clinical.loc[0, "label"]]
        pet_image = torch.unsqueeze(normalized(pet_image), 0)
        ct_image = torch.unsqueeze(normalized(ct_image), 0)
        image = torch.cat((ct_image, pet_image))
        return image, seg_image, age/100, sex, label    # age normalization

    def __len__(self):
        return len(self.study_list)

start = time.time()
classes = pd.read_csv(r'E:\dataset\target.csv', index_col=0)
clinical = pd.read_csv(r'E:\dataset\Clinical Metadata FDG PET_CT Lesions.csv')
test_data = MyData(clinical, classes, [-1])
print("finish test data")
lr_list = [1e-5, 2e-5, 4e-5, 1e-4, 1e-3]
weight_decay_list = [1e-5, 2e-5, 4e-5, 1e-4, 1e-3]
batch = 4
ModelList = []
seg_loss = loss.DiceLoss()
cls_loss = nn.CrossEntropyLoss()
test_dataloader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True, drop_last=False)
TrainModelPerformance = pd.DataFrame(columns=[1e-5, 2e-5, 4e-5, 1e-4, 1e-3], index=[1e-5, 2e-5, 4e-5, 1e-4, 1e-3])
ValModelPerformance = pd.DataFrame(columns=[1e-5, 2e-5, 4e-5, 1e-4, 1e-3], index=[1e-5, 2e-5, 4e-5, 1e-4, 1e-3])

for lr in lr_list:
    for weight_decay in weight_decay_list:
        trainLossList = []
        ValLossList = []
        for validation in range(10):
            print(f"start training {validation}")
            trainlist = list(range(validation))
            trainlist.extend(list(range(validation+1, 10)))
            train_data = MyData(clinical, classes, trainlist)
            train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False)
            valid_data = MyData(clinical, classes, [validation])
            valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch, shuffle=True, drop_last=False)

            model = UNet3D()
            # model = VNet()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)    # L2 loss

            model, TrainLoss = training(model, optimizer, seg_loss, cls_loss, train_dataloader)
            trainLossList.append(TrainLoss)

            ValLoss = val(model, valid_dataloader, seg_loss, cls_loss)
            ValLossList.append(ValLoss)

        avg_val_loss = np.mean(ValLossList)
        TrainModelPerformance.loc[lr, weight_decay] = np.mean(trainLossList)
        ValModelPerformance.loc[lr, weight_decay] = np.mean(trainLossList)
TrainModelPerformance.to_csv("UNet Performance/TrainModelPerformance.csv")
ValModelPerformance.to_csv("UNet Performance/ValModelPerformance.csv")


# retrain for best 
lr, weight_decay = ValModelPerformance.stack().idxmax()
train_data = MyData(clinical, classes, list(range(10)))
train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False)

model = UNet3D()
# model = VNet()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)    # L2 loss

model, _, trainLossList = training(model, optimizer, seg_loss, cls_loss, train_dataloader, verbose=True)
TestLoss = val(model, test_dataloader, seg_loss, cls_loss)
trainLossList.to_csv("UNet Performance/trainLoss.csv")
print("-"*20, f"\nTest Loss: {TestLoss: 0.5f}")
print(f"total time: {time.time() - start: .2f}s")
