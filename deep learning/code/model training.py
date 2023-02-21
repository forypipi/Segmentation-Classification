import time

import numpy as np
import torch
from torch import nn, Tensor, optim, tensor
import torch.nn.functional as F
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

def training(model, optimizer, seg_loss, cls_loss, train_loader: DataLoader, verbose=False, EPOCHS=1, alpha=0.5):
    # log_steps = 1
    # global_step = 0
    # # start = time.time()

    if verbose:
        loss_list, seg_loss_list, cls_loss_list = [], [], []
    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, total=len(train_loader))
        for data in loop:
            images, seg_images, ages, sexs, labels = data

            # image: [从头向下, 从前向后, 从左向右]
            optimizer.zero_grad()
            segmentation, classification = model(images, sexs, ages)

            # 损失函数采用nn.CrossEntropyLoss()
            # if labels == 0:
            #     train_loss = cls_loss(classification, labels)
            # else:

            SegLoss = seg_loss(segmentation, seg_images)
            ClsLoss = cls_loss(classification, labels)
            train_loss = alpha * SegLoss + (1 - alpha) * ClsLoss
            if verbose:
                loss_list.append(float(train_loss.item()))
                seg_loss_list.append(float(SegLoss.item()))
                cls_loss_list.append(float(ClsLoss.item()))

            train_loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{epoch+1}/{EPOCHS}]')
            loop.set_postfix(loss=train_loss.item(), seg_loss=SegLoss.item(), cls_loss=ClsLoss.item(), label=labels.item())

            # 每100个batch检查一次交叉熵的值
            # global_step += 1
            # if global_step % log_steps == 0:
            #     print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f s/step"
            #           % (global_step, epoch, step, float(train_loss), (time.time() - start) / global_step))

    if verbose:
        return model, train_loss, pd.DataFrame(loss_list), pd.DataFrame(seg_loss_list), pd.DataFrame(cls_loss_list)
    else:
        return model, train_loss


def val(model, val_loader: DataLoader, seg_loss, cls_loss):
    model.eval()
    seg_total_loss, cls_total_loss = 0, 0
    loop = tqdm(val_loader, total=len(val_loader))

    with torch.no_grad():
        for data in loop:
            images, seg_images, ages, sexs, labels = data
            segmentation, classification = model(images, sexs, ages)
            seg_total_loss += seg_loss(segmentation, seg_images)
            cls_total_loss += cls_loss(classification, labels)
        avg_seg_loss = seg_total_loss / len(val_loader)
        avg_cls_loss = cls_total_loss / len(val_loader)
    return avg_seg_loss, avg_cls_loss

class MyData(Dataset):
    def __init__(self, clinical: pd.DataFrame, data_label: pd.DataFrame, label_list: list, device='CPU', root_dir=r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'):
        self.clinical = clinical
        self.data_label = data_label
        self.study_list = self.data_label[self.data_label["training_label"].isin(label_list)].reset_index().loc[:, 'Study UID']
        self.root_dir = root_dir
        self.device = device

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
        seg_image = Tensor(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)))
        clinical = pd.read_csv(os.path.join(patient_dir, "clinical.csv"))
        age = tensor(int(clinical.loc[0, "age"])).to(self.device)
        sex = clinical.loc[0, "sex"]
        if isinstance(sex, str):
            sex = sex_dict[sex]
        else:
            sex = 2
        sex = F.one_hot(tensor(sex), num_classes=3).to(self.device)
        label = tensor(label_dict[clinical.loc[0, "label"]]).to(self.device)
        pet_image = torch.unsqueeze(normalized(pet_image), 0)
        ct_image = torch.unsqueeze(normalized(ct_image), 0)
        image = torch.cat((ct_image, pet_image))
        return image.to(self.device), seg_image.to(self.device), age/100, sex, label    # age normalization

    def __len__(self):
        return len(self.study_list)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("running on GPU")
else:
    device = torch.device('cpu')
    print("running on CPU")


start = time.time()
root_dir = r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'
classes = pd.read_csv(os.path.join(root_dir, 'split_train_val_test.csv'), index_col=0)
clinical = pd.read_csv(os.path.join(root_dir, 'Clinical Metadata FDG PET_CT Lesions.csv'))
if hasattr(torch.cuda, 'empty_cache'):
	torch.cuda.empty_cache()
lr = 1e-5
alpha_list = [0.5]
alpha_list.extend(np.arange(0, 1.1, 0.2).tolist())
weight_decay = 1e-5
batch = 1
ModelList = []
seg_loss = loss.DiceLoss().to(device)
cls_loss = nn.CrossEntropyLoss().to(device)
test_data = MyData(clinical, classes, [-1], device)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True, drop_last=False)
TrainModelPerformance = pd.DataFrame(columns=alpha_list, index=[0])
ValModelPerformance = pd.DataFrame(columns=alpha_list, index=[0])

for alpha in alpha_list:
    trainLossList, ValLossList = [], []
    for validation in range(3):
        print(f"start training on valid {validation}")
        trainlist = list(range(validation))
        trainlist.extend(list(range(validation+1, 3)))
        train_data = MyData(clinical, classes, trainlist, device)
        train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False)
        valid_data = MyData(clinical, classes, [validation], device)
        valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch, shuffle=True, drop_last=False)

        model = UNet3D().to(device)
        # model = VNet().to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)    # L2 loss

        model, TrainLoss = training(model, optimizer, seg_loss, cls_loss, train_dataloader, alpha=alpha)
        trainLossList.append(TrainLoss.cpu().detach())

        print(f"start validation on valid {validation}")

        avg_seg_loss, avg_cls_loss = val(model, valid_dataloader, seg_loss, cls_loss)
        ValLoss = alpha * avg_seg_loss + (1 - alpha) * avg_cls_loss
        ValLossList.append(ValLoss.cpu())
    TrainModelPerformance.loc[0, alpha] = np.mean(trainLossList)
    ValModelPerformance.loc[0, alpha] = np.mean(ValLossList)

TrainModelPerformance.to_csv("./Performance/TrainModelPerformance.csv")
ValModelPerformance.to_csv("./Performance/ValModelPerformance.csv")

# retrain for best 
_, alpha = ValModelPerformance.stack().astype(float).idxmax()
train_data = MyData(clinical, classes, list(range(10)), device=device)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False)

model = UNet3D().to(device)
# model = VNet()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)    # L2 loss

model, _, trainLossList, trainSegLossList, trainClsLossList = training(model, optimizer, seg_loss, cls_loss, train_dataloader, verbose=True, alpha=alpha)
SegLoss, ClsLoss = val(model, test_dataloader, seg_loss, cls_loss)
torch.save(model, r'./model/UNet3D.pt')
trainLossList.to_csv("./Performance/trainLoss.csv")
print("-"*20, f"\nTest total Loss: {(SegLoss+ClsLoss)/2}, Test Seg Loss: {SegLoss: 0.5f}, Test Cls Loss: {ClsLoss: 0.5f}")
print(f"total time: {time.time() - start: .2f}s")
