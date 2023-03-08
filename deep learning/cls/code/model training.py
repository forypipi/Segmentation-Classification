import time
import multiprocessing as mp

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn, Tensor, optim, tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, dataset

import SimpleITK as sitk
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

import loss
from models.resnet import ResNet18
from models.densenet import DenseNet121_3d
from models.models import UNet

def split_train_val_test(clinical: pd.DataFrame, test_radio=0.1, kFold=3):
    train_set, _ = train_test_split(clinical, test_size=test_radio, stratify=clinical['class'], random_state=42)
    train_set.insert(clinical.shape[1], 'training_label', np.nan)
    X, y = train_set['name'], train_set['class']
    skf = StratifiedKFold(n_splits=kFold, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(skf.split(X, y)):
        print('*' * 20, f'{fold + 1}', '*' * 20)
        train_set.iloc[val_idx, 2] = [fold] * len(val_idx)
    train_set = train_set.drop("class", axis=1)
    clinical = clinical.merge(train_set, how='left', left_on='name', right_on='name')
    clinical['training_label'].fillna(-1, inplace=True)
    clinical['training_label'] = clinical['training_label'].astype('int')
    clinical.to_csv("./cls/data/training label.csv")
    return clinical

def acc(pred, true, writer: SummaryWriter, epoch):
    accuracy = np.round(accuracy_score(true, pred), 4)
    writer.add_scalar("accuracy", accuracy, epoch)
    return accuracy

# def ReadDcm(path):
    # """
    # :param path: folder path
    # :return: image object, reader object(read metadata)
    # """
    # reader = sitk.ImageSeriesReader()
    # # dicom_names = reader.GetGDCMSeriesFileNames(path)
    # # print(dicom_names)
    # reader.SetFileNames(path)
    # reader.MetaDataDictionaryArrayUpdateOn()
    # reader.LoadPrivateTagsOn()
    # image = reader.Execute()
    # return sitk.GetArrayFromImage(image), reader

def normalized(img, range1=(0, 1), device="cpu"):
    img = img.astype(np.float32)      # int32 to float32
    if img.max() != img.min():
        img = (img - img.min()) / (img.max() - img.min())    # to 0--1
    img = img * (range1[1] - range1[0]) + range1[0]
    return torch.as_tensor(img, dtype=torch.float32, device=device)

def training(model: nn.modules, optimizer, cls_loss, train_loader: DataLoader, test_loader: DataLoader, writer: dict, verbose=False, EPOCHS=50, tag="val0"):
    model.train()
    with torch.enable_grad():
        if verbose:
            loss_list = []

        avg_train_loss = 0
        step = 0
        for epoch in range(EPOCHS):

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            loop = tqdm(train_loader, total=len(train_loader))
            for data in loop:
                step += 1
                images, labels, _ = data

                # image: [从头向下, 从前向后, 从左向右]
                optimizer.zero_grad()
                # classification = model(images, sexs, ages)
                classification = model(images)
                train_loss = cls_loss(classification, labels)
                if verbose:
                    loss_list.append(float(train_loss.cpu()))
                avg_train_loss += float(train_loss.cpu())

                train_loss.backward()
                optimizer.step()

                if step % 5 == 0:
                    writer["train"].add_scalar("loss", avg_train_loss / 5, step)
                    avg_train_loss = 0
                # writer.add_scalar("train loss", train_loss, step)

                loop.set_description(f'Epoch [{epoch + 1}/{EPOCHS}]')
                loop.set_postfix(loss=float(train_loss.cpu()), label=list(map(int, labels.cpu())))
                # loop.set_postfix(loss=float(train_loss.cpu()))

            val(model, test_loader, cls_loss, writer["test"], (epoch+1)*len(loop), tag=tag)

    if verbose:
        return model, train_loss, pd.DataFrame(loss_list)
    else:
        return model, train_loss

def val(model, data_loader: DataLoader, cls_loss, writer=None, epoch=50, tag="val0"):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        loop = tqdm(data_loader, total=len(data_loader))
        step = 0
        pred_result, true_result, pred_prob, study = [], [], [], []
        for data in loop:
            step += 1
            images, labels, study_id = data
            # classification = model(images, sexs, ages)
            classification = model(images)
            clsloss = cls_loss(classification, labels)
            classification = nn.Softmax(dim=1)(classification)
            pred_result.extend(list(map(int, torch.argmax(classification, dim=1).cpu())))
            pred_prob.extend(classification.cpu().tolist())
            true_result.extend(list(map(int, labels.cpu())))
            study.extend(study_id)
            total_loss += clsloss
        avg_cls_loss = total_loss / len(data_loader)
        if writer != None:
            writer.add_scalar("loss", avg_cls_loss, epoch)
            acc(pred_result, true_result, writer=writer, epoch=epoch)
    result = pd.DataFrame({"pred": pred_result, "true": true_result, "pred prob": pred_prob}, index=study)
    result.to_csv(f"./cls/data/output/ResNet3D/{tag}/test_result_{epoch}.csv")
    return avg_cls_loss, result

class MyData(Dataset):
    def __init__(self, data_label: pd.DataFrame, label_list: list, device='CPU', root_dir=r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'):
        self.data = data_label[data_label['training_label'].isin(label_list)]
        self.data_label = data_label
        self.root_dir = root_dir
        self.device = device

    def __getitem__(self, item):
        # sex_dict = {'F': 0, 'M': 1}
        label_dict = {'NEGATIVE': 0, 'LYMPHOMA': 1, 'MELANOMA': 2, 'LUNG_CANCER': 3}
        patient_id = self.data.iloc[item, 1]
        diagnose = self.data.iloc[item, 0]
        patient_dir = os.path.join(self.root_dir, diagnose, patient_id)
        ct_path = os.path.join(patient_dir, 'CT_nii', '1.nii')
        # pet_path = os.path.join(patient_dir, 'PET_nii', '1.nii')
        ct_image = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
        # pet_image = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))
        clinical = pd.read_csv(os.path.join(patient_dir, "clinical.csv"))
        # age = tensor(int(clinical.loc[0, "age"])).to(self.device)
        # sex = clinical.loc[0, "sex"]
        # if isinstance(sex, str):
        #     sex = sex_dict[sex]
        # else:
        #     sex = 2
        # sex = F.one_hot(tensor(sex), num_classes=3).to(self.device)
        label = tensor(label_dict[clinical.loc[0, "label"]]).to(self.device)
        # pet_image = torch.unsqueeze(normalized(pet_image), 0)
        ct_image = torch.unsqueeze(normalized(ct_image, device=self.device), 0)
        # image = torch.cat((ct_image, pet_image))
        # return image.to(self.device), label, patient_id
        # return pet_image.to(self.device), label, patient_id
        return ct_image.to(self.device), label, patient_id


    def __len__(self):
        return len(self.data)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("running on GPU")
else:
    device = torch.device('cpu')
    print("running on CPU")


start = time.time()

root_dir = r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'
# root_dir = r'/data/zhxie/oufu_data_400G/preprocessed'
# root_dir = r'E:\dataset\preprocessed'

kFold = 3
labels = ["NEGATIVE", "LUNG_CANCER", "LYMPHOMA", "MELANOMA"]
dfs = [pd.DataFrame({"class": [label] * len(os.listdir(os.path.join(root_dir, label))), "name": os.listdir(os.path.join(root_dir, label))}) for label in labels]
Whole_df = pd.concat(dfs)
classes = split_train_val_test(clinical=Whole_df, kFold=kFold, test_radio=0.2)

model_name = "ResNet3D"
lr = 5e-4
torch.set_default_dtype(torch.float32)
weight_decay = 5e-2
batch = 16
eopchs = 20
# seg_loss = sigmoid_focal_loss().to(device)
seg_loss = loss.WeightedFocalLoss(device=device)
cls_loss = nn.CrossEntropyLoss().to(device)

test_data = MyData(classes, [-1], device, root_dir=root_dir)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch, shuffle=False, drop_last=False)
trainLossList, ValLossList = [], []
for validation in range(kFold):
    print(f"start training on valid {validation}")
    
    paths = [
        os.path.join('./cls/log/train', model_name, "val"+str(validation)),
        os.path.join('./cls/log/test', model_name, "val"+str(validation)),
        os.path.join("./cls/Performance", model_name, "val"+str(validation)),
        os.path.join("./cls/data/output", model_name, "val"+str(validation))
    ]
    for dir in paths:
        if not os.path.exists(dir):
            os.makedirs(dir)
    train_writer = SummaryWriter(os.path.join('./cls/log/train', model_name, "val"+str(validation)))
    test_writer = SummaryWriter(os.path.join('./cls/log/test', model_name, "val"+str(validation)))

    trainlist = list(range(validation))
    trainlist.extend(list(range(validation+1, kFold)))
    train_data = MyData(classes, trainlist, device, root_dir=root_dir)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False)
    valid_data = MyData(classes, [validation], device, root_dir=root_dir)
    valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch, shuffle=False, drop_last=False)

    # model = UNet().to(device)
    # model = models.VNet().to(device)
    model = ResNet18().to(device)
    # model = DenseNet121_3d().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)    # L2 loss

    model, _, trainLossList = training(model, optimizer, cls_loss, train_dataloader, valid_dataloader, verbose=True, writer={"train": train_writer, "test": test_writer}, tag="val"+str(validation), EPOCHS=eopchs)
    print(f"start validation on valid {validation}")
    testLoss, test_result = val(model, test_dataloader, cls_loss)
    print("get training result")
    testLoss, train_result = val(model, train_dataloader, cls_loss)

    torch.save(model, os.path.join('./cls/model', model_name+str(validation)+'.pt'))
    trainLossList.to_csv(os.path.join("./cls/Performance", model_name, "val"+str(validation), "trainLoss.csv"))
    test_result.to_csv(os.path.join("./cls/data/output", model_name, "val"+str(validation), "test_result.csv"))
    train_result.to_csv(os.path.join("./cls/data/output", model_name, "val"+str(validation), "train_result.csv"))

# loss_df = pd.DataFrame({"train loss": trainLossList, "valid loss": ValLossList})
# loss_df.to_csv("./cls/Performance/UNet/search loss.csv")


# retrain for best
# train_writer = SummaryWriter(os.path.join('./cls/log/train', model_name, "val"))
# test_writer = SummaryWriter(os.path.join('./cls/log/test', model_name, "val"))
# train_data = MyData(classes, list(range(kFold)), device=device, root_dir=root_dir)
# train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False)

# # model = UNet().to(device)
# # model = models.VNet().to(device)
# model = ResNet18().to(device)
# # model = DenseNet121_3d().to(device)
# optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)    # L2 loss

# model, _, trainLossList = training(model, optimizer, cls_loss, train_dataloader, test_dataloader, verbose=True, writer={"train": train_writer, "test": test_writer}, EPOCHS=eopchs)
# print("get test result")
# testLoss, test_result = val(model, test_dataloader, cls_loss)
# print("get train result")
# trainLoss, train_result = val(model, train_dataloader, cls_loss)

# torch.save(model, os.path.join('./cls/model', model_name+'.pt'))
# trainLossList.to_csv(os.path.join("./cls/Performance", model_name, "trainLoss.csv"))
# test_result.to_csv(os.path.join("./cls/data/output", model_name, "test_result.csv"))
# train_result.to_csv(os.path.join("./cls/data/output", model_name, "train_result.csv"))
# print("-"*20, f"\nTest total Loss: {testLoss}")
# # writer.close()
print(f"total time: {time.time() - start: .2f}s")