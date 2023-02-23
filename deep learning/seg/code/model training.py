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

def training(model, optimizer, seg_loss, cls_loss, train_loader: DataLoader, writer, verbose=False, EPOCHS=1, alpha=0.5, mode='both'):
    # log_steps = 1
    # global_step = 0
    # # start = time.time()
    step = 0
    avg_train_loss, avg_SegLoss, avg_ClsLoss = 0, 0, 0
    if mode == "both":
        if verbose:
            loss_list, seg_loss_list, cls_loss_list = [], [], []
        for epoch in range(EPOCHS):
            if hasattr(torch.cuda, 'empty_cache'):
	            torch.cuda.empty_cache()
            loop = tqdm(train_loader, total=len(train_loader))
            for data in loop:
                step += 1
                images, seg_images, ages, sexs, labels = data

                # image: [从头向下, 从前向后, 从左向右]
                optimizer.zero_grad()
                segmentation, classification = model(images, sexs, ages)
                seg_images = seg_images.reshape((-1, 1, 256, 256, 256))
                SegLoss = seg_loss(segmentation, seg_images)
                # SegLoss = sigmoid_focal_loss(segmentation, seg_images, reduction='mean').to(device)
                ClsLoss = cls_loss(classification, labels)
                train_loss = alpha * SegLoss + (1 - alpha) * ClsLoss
                # if verbose:
                #     loss_list.append(float(train_loss.item()))
                #     seg_loss_list.append(float(SegLoss.item()))
                #     cls_loss_list.append(float(ClsLoss.item()))
                avg_train_loss += train_loss
                avg_SegLoss += SegLoss
                avg_ClsLoss += ClsLoss
                train_loss.backward()
                optimizer.step()
                if step % 5 == 0:
                    writer.add_scalar("train loss", avg_train_loss / 5, step)
                    writer.add_scalar("cls loss", avg_SegLoss / 5, step)
                    writer.add_scalar("seg loss", avg_ClsLoss / 5, step)
                    avg_train_loss = 0
                    avg_SegLoss = 0
                    avg_ClsLoss = 0

                loop.set_description(f'Epoch [{epoch+1}/{EPOCHS}]')
                loop.set_postfix(loss=train_loss.item(), seg_loss=SegLoss.item(), cls_loss=ClsLoss.item(), label=list(map(int, labels.cpu())))
        if verbose:
            return model, train_loss, pd.DataFrame(loss_list), pd.DataFrame(seg_loss_list), pd.DataFrame(cls_loss_list)
        else:
            return model, train_loss
    elif mode == "seg":
        if verbose:
            loss_list, seg_loss_list, cls_loss_list = [], [], []
        for epoch in range(EPOCHS):
            if hasattr(torch.cuda, 'empty_cache'):
	            torch.cuda.empty_cache()
            loop = tqdm(train_loader, total=len(train_loader))
            for data in loop:
                step += 1
                images, seg_images, ages, sexs, labels = data

                # image: [从头向下, 从前向后, 从左向右]
                optimizer.zero_grad()
                segmentation, _ = model(images, sexs, ages)
                seg_images = seg_images.reshape((-1, 1, 256, 256, 256))
                SegLoss = seg_loss(segmentation, seg_images)
                # SegLoss = sigmoid_focal_loss(segmentation, seg_images, reduction='mean').to(device)
                train_loss = SegLoss
                train_loss.requires_grad_(True)
                if verbose:
                    loss_list.append(float(train_loss.item()))
                    seg_loss_list.append(float(SegLoss.item()))

                avg_train_loss += train_loss
                avg_SegLoss += SegLoss
                train_loss.backward()
                optimizer.step()
                if step % 5 == 0:
                    writer.add_scalar("train loss", avg_train_loss / 5, step)
                    writer.add_scalar("seg loss", avg_ClsLoss / 5, step)
                    avg_train_loss = 0
                    avg_SegLoss = 0

                loop.set_description(f'Epoch [{epoch + 1}/{EPOCHS}]')
                loop.set_postfix(loss=train_loss.item(), seg_loss=SegLoss.item(), label=list(map(int, labels.cpu())))
        if verbose:
            return model, train_loss, pd.DataFrame(loss_list), pd.DataFrame(seg_loss_list)
        else:
            return model, train_loss
    elif mode == "cls":
        if verbose:
            loss_list, seg_loss_list, cls_loss_list = [], [], []
        for epoch in range(EPOCHS):
            if hasattr(torch.cuda, 'empty_cache'):
            	torch.cuda.empty_cache()

            loop = tqdm(train_loader, total=len(train_loader))
            for data in loop:
                step += 1
                images, seg_images, ages, sexs, labels = data

                # image: [从头向下, 从前向后, 从左向右]
                optimizer.zero_grad()
                _, classification = model(images, sexs, ages)
                ClsLoss = cls_loss(classification, labels)
                train_loss = ClsLoss
                if verbose:
                    loss_list.append(float(train_loss.item()))
                    cls_loss_list.append(float(ClsLoss.item()))

                avg_train_loss += train_loss
                train_loss.backward()
                optimizer.step()
                if step % 5 == 0:
                    writer.add_scalar("train loss", avg_train_loss / 5, step)
                    avg_train_loss = 0

        if verbose:
            return model, train_loss, pd.DataFrame(loss_list), pd.DataFrame(seg_loss_list), pd.DataFrame(cls_loss_list)
        else:
            return model, train_loss
    else:
        raise Exception("wrong input: mode, should be one of 'both', 'seg', 'cls'")

def val(model, val_loader: DataLoader, seg_loss, cls_loss, writer, mode="both"):
    model.eval()
    seg_total_loss, cls_total_loss = 0, 0
    loop = tqdm(val_loader, total=len(val_loader))
    step = 0
    with torch.no_grad():
        if mode == "both":
            for data in loop:
                step += 1
                images, seg_images, ages, sexs, labels = data
                segmentation, classification = model(images, sexs, ages)
                segmentation = nn.Sigmoid()(segmentation)
                segloss = seg_loss(segmentation, seg_images)
                seg_total_loss += segloss
                # seg_total_loss += sigmoid_focal_loss(segmentation, seg_images).to(device)
                clsloss = cls_loss(classification, labels)
                cls_total_loss += clsloss
                writer.add_scalar("test seg loss", segloss, step)
                writer.add_scalar("test cls loss", clsloss, step)

            avg_seg_loss = seg_total_loss / len(val_loader)
            avg_cls_loss = cls_total_loss / len(val_loader)
        elif mode == "seg":
            for data in loop:
                step += 1
                images, seg_images, ages, sexs, labels = data
                segmentation, _ = model(images, sexs, ages)
                segmentation = nn.Sigmoid()(segmentation)

                segloss = seg_loss(segmentation, seg_images)
                seg_total_loss += segloss
                writer.add_scalar("test seg loss", segloss, step)

            avg_seg_loss = seg_total_loss / len(val_loader)
            avg_cls_loss = None
        elif mode == "cls":
            for data in loop:
                images, seg_images, ages, sexs, labels = data
                _, classification = model(images, sexs, ages)
                classification = nn.Softmax()(classification)

                clsloss = cls_loss(classification, labels)
                cls_total_loss += clsloss
                writer.add_scalar("test cls loss", clsloss, step)
                
            avg_seg_loss = None
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
        seg_image = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        seg_image[seg_image==255] = 1
        seg_image = Tensor(seg_image)
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
# root_dir = r'E:\dataset\preprocessed'
classes = pd.read_csv(os.path.join(root_dir, 'split_train_val_test.csv'), index_col=0)
clinical = pd.read_csv(os.path.join(root_dir, 'Clinical Metadata FDG PET_CT Lesions.csv'))

writer = SummaryWriter('./log/UNet/cls')
mode = "cls"
lr = 1e-5
# alpha_list = [0,25, 0.5, 0.75]
alpha_list = [0.5]

weight_decay = 1e-5
batch = 4
# seg_loss = sigmoid_focal_loss().to(device)
seg_loss = loss.WeightedFocalLoss(device=device)
cls_loss = nn.CrossEntropyLoss().to(device)
test_data = MyData(clinical, classes, [-1], device, root_dir=root_dir)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True, drop_last=False)
TrainModelPerformance = pd.DataFrame(columns=alpha_list, index=[0])
ValModelPerformance = pd.DataFrame(columns=alpha_list, index=[0])

for alpha in alpha_list:
    print(f"training alpha={alpha}")
    trainLossList, ValLossList = [], []
    for validation in range(1):
        print(f"start training on valid {validation}")
        trainlist = list(range(validation))
        trainlist.extend(list(range(validation+1, 3)))
        train_data = MyData(clinical, classes, trainlist, device, root_dir=root_dir)
        train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False)
        valid_data = MyData(clinical, classes, [validation], device, root_dir=root_dir)
        valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch, shuffle=True, drop_last=False)

        model = models.UNet3D_cls().to(device)
        # model = models.UNet3D_seg().to(device)
        # model = models.UNet3D().to(device)

        # model = models.VNet().to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)    # L2 loss

        model, TrainLoss = training(model, optimizer, seg_loss, cls_loss, train_dataloader, writer=writer, alpha=alpha, mode=mode)
        trainLossList.append(TrainLoss.cpu().detach())

        print(f"start validation on valid {validation}")

        avg_seg_loss, avg_cls_loss = val(model, valid_dataloader, seg_loss, cls_loss, writer=writer, mode=mode)
        ValLoss = alpha * avg_seg_loss + (1 - alpha) * avg_cls_loss
        ValLossList.append(ValLoss.cpu())
    TrainModelPerformance.loc[0, alpha] = np.mean(trainLossList)
    ValModelPerformance.loc[0, alpha] = np.mean(ValLossList)

TrainModelPerformance.to_csv("./Performance/UNet_seg&cls/TrainModelPerformance.csv")
ValModelPerformance.to_csv("./Performance/UNet_seg&cls/ValModelPerformance.csv")

# retrain for best
_, alpha = ValModelPerformance.stack().astype(float).idxmax()
train_data = MyData(clinical, classes, list(range(10)), device=device)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False)

model = models.UNet3D_cls().to(device)
# model = VNet()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)    # L2 loss

model, _, trainLossList, trainSegLossList, trainClsLossList = training(model, optimizer, seg_loss, cls_loss, train_dataloader, verbose=True, alpha=alpha, mode=mode)
SegLoss, ClsLoss = val(model, test_dataloader, seg_loss, cls_loss, mode=mode)
torch.save(model, r'./model/UNet3D.pt')
trainLossList.to_csv("./Performance/UNet_seg&cls/trainLoss.csv")
print("-"*20, f"\nTest total Loss: {(SegLoss+ClsLoss)/2}, Test Seg Loss: {SegLoss: 0.5f}, Test Cls Loss: {ClsLoss: 0.5f}")
print(f"best alpha: {alpha}")
writer.close()
print(f"total time: {time.time() - start: .2f}s")