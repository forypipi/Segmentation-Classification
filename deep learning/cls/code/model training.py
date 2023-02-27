import time

import numpy as np
import torch
from torch import nn, Tensor, optim, tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import loss
import models

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

def training(model, optimizer, cls_loss, train_loader: DataLoader, writer, verbose=False, EPOCHS=75):
    step = 0
    avg_train_loss = 0
    if verbose:
        loss_list = []
    for epoch in range(EPOCHS):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        loop = tqdm(train_loader, total=len(train_loader))
        for data in loop:
            step += 1
            images, ages, sexs, labels, _ = data

            # image: [从头向下, 从前向后, 从左向右]
            optimizer.zero_grad()
            classification = model(images, sexs, ages)
            train_loss = cls_loss(classification, labels)
            if verbose:
                loss_list.append(float(train_loss.cpu()))
            avg_train_loss += train_loss
            train_loss.backward()
            optimizer.step()
            if step % 5 == 0:
                writer.add_scalar("train loss", avg_train_loss / 5, step)
                avg_train_loss = 0
            loop.set_description(f'Epoch [{epoch + 1}/{EPOCHS}]')
            loop.set_postfix(loss=float(train_loss.cpu()), label=list(map(int, labels.cpu())))

    if verbose:
        return model, train_loss, pd.DataFrame(loss_list)
    else:
        return model, train_loss

def val(model, val_loader: DataLoader, cls_loss, writer):
    model.eval()
    total_loss = 0
    loop = tqdm(val_loader, total=len(val_loader))
    step = 0
    pred_result, true_result, pred_prob, study = [], [], [], []
    with torch.no_grad():
        for data in loop:
            step += 1
            images, ages, sexs, labels, study_id = data
            classification = model(images, sexs, ages)
            classification = nn.Softmax(dim=1)(classification)
            pred_result.extend(list(map(int, torch.argmax(classification, dim=1).cpu())))
            pred_prob.extend(classification.cpu().tolist())
            true_result.extend(list(map(int, labels.cpu())))
            study.extend(study_id)
            clsloss = cls_loss(classification, labels)
            total_loss += clsloss
            writer.add_scalar("test loss", total_loss / step, step)
        avg_cls_loss = total_loss / len(val_loader)
    result = pd.DataFrame({"pred": pred_result, "true": true_result, "pred prob": pred_prob}, index=study)
    return avg_cls_loss, result

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
        ct_image = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
        pet_image = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))
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
        return image.to(self.device), age/100, sex, label, study_id    # age normalization

    def __len__(self):
        return len(self.study_list)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("running on GPU")
else:
    device = torch.device('cpu')
    print("running on CPU")


start = time.time()
# root_dir = r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'
root_dir = r'/data/zhxie/oufu_data_400G/preprocessed'
# root_dir = r'E:\dataset\preprocessed'
classes = pd.read_csv(os.path.join(root_dir, 'split_train_val_test.csv'), index_col=0)
clinical = pd.read_csv(os.path.join(root_dir, 'Clinical Metadata FDG PET_CT Lesions.csv'))

writer = SummaryWriter('./cls/log/VNet')
lr = 1e-5

weight_decay = 1e-5
batch = 16
# seg_loss = sigmoid_focal_loss().to(device)
seg_loss = loss.WeightedFocalLoss(device=device)
cls_loss = nn.CrossEntropyLoss().to(device)
test_data = MyData(clinical, classes, [-1], device, root_dir=root_dir)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True, drop_last=False)

trainLossList, ValLossList = [], []
# for validation in range(3):
#     print(f"start training on valid {validation}")
#     trainlist = list(range(validation))
#     trainlist.extend(list(range(validation+1, 3)))
#     train_data = MyData(clinical, classes, trainlist, device, root_dir=root_dir)
#     train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False)
#     valid_data = MyData(clinical, classes, [validation], device, root_dir=root_dir)
#     valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch, shuffle=True, drop_last=False)

#     model = models.UNet().to(device)
#     # model = models.VNet().to(device)

#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)    # L2 loss

#     model, TrainLoss = training(model, optimizer, cls_loss, train_dataloader, writer=writer)
#     trainLossList.append(float(TrainLoss.cpu().detach()))

#     print(f"start validation on valid {validation}")

#     avg_cls_loss, _ = val(model, valid_dataloader, cls_loss, writer=writer)
#     ValLossList.append(float(avg_cls_loss.cpu()))

# loss_df = pd.DataFrame({"train loss": trainLossList, "valid loss": ValLossList})
# loss_df.to_csv("./cls/Performance/UNet/search loss.csv")

# retrain for best
train_data = MyData(clinical, classes, list(range(10)), device=device, root_dir=root_dir)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False)

# model = models.UNet().to(device)
model = models.VNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)    # L2 loss

model, _, trainLossList = training(model, optimizer, cls_loss, train_dataloader, verbose=True, writer=writer)

# model = torch.load("./cls/model/VNet.pt")
testLoss, result = val(model, test_dataloader, cls_loss, writer=writer)

torch.save(model, r'./cls/model/VNet.pt')
trainLossList.to_csv("./cls/Performance/VNet/trainLoss.csv")
result.to_csv("./cls/data/output/VNet/result.csv")
print("-"*20, f"\nTest total Loss: {testLoss}")
# writer.close()
print(f"total time: {time.time() - start: .2f}s")