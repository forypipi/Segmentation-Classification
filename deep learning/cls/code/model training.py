import pickle
import time
import multiprocessing as mp
import itertools

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn, Tensor, optim, tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, dataset

import SimpleITK as sitk
import nibabel as nib
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

import loss
from models.ViT import ViT
from models.resnet import ResNet18
from models.densenet import DenseNet121_3d
from models.models import UNet
from models.swinTransformer import SwinTransformer

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
    if writer != None:
        writer.add_scalar("accuracy", accuracy, epoch)
    return accuracy

def normalized(img, range1=(0, 1), device="cuda"):
    img = torch.as_tensor(img, dtype=torch.float32, device=device)      # int32 to float32
    if img.max() != img.min():
        img = (img - img.min()) / (img.max() - img.min())    # to 0--1
    img = img * (range1[1] - range1[0]) + range1[0]
    return img

def training(model: nn.modules, optimizer, cls_loss, train_loader: DataLoader, test_loader: DataLoader, writer: dict, verbose=False, EPOCHS=50, threshold=0.1, tag="val0"):
    model.train()
    with torch.enable_grad():
        if verbose:
            loss_list = []
        early_stop = 0
        step = 0
        max_acc = 0
        for epoch in range(int(EPOCHS)):
            epo_loss_list = []

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            loop = tqdm(train_loader, total=len(train_loader))
            for data in loop:
                step += 1
                images, labels = data
                # image: [从头向下, 从前向后, 从左向右]
                optimizer.zero_grad()
                # classification = model(images, sexs, ages)
                classification = model(images)
                train_loss = torch.abs(cls_loss(classification, labels) - threshold) + threshold
                epo_loss_list.append(float(train_loss.cpu()))

                train_loss.backward()
                optimizer.step()

                if step % 5 == 0 and len(epo_loss_list) >= 5:
                    writer["train"].add_scalar("loss", sum(epo_loss_list[-5:]) / 5, step)
                # writer.add_scalar("train loss", train_loss, step)

                loop.set_description(f'Epoch [{epoch + 1}/{EPOCHS}]')
                # loop.set_postfix(loss=float(train_loss.cpu()))

            _, accuracy = val(model, test_loader, cls_loss, writer["test"], (epoch+1)*len(loop), tag=tag)

            if verbose:
                loss_list.extend(epo_loss_list)
            
            if max_acc < accuracy:
                max_acc = accuracy
                early_stop = epoch

    if verbose:
        return early_stop, model, train_loss, pd.DataFrame(loss_list), max_acc
    else:
        return early_stop, model, train_loss, max_acc

def val(model, data_loader: DataLoader, cls_loss, writer=None, epoch=50, tag="val0", device="cuda"):
    model.eval()
    with torch.no_grad():
        total_loss = torch.as_tensor(0., device=device)
        loop = tqdm(data_loader, total=len(data_loader))
        pred_result, true_result, pred_prob = torch.as_tensor([], device=device), torch.as_tensor([], device=device), torch.as_tensor([], device=device)
        for data in loop:
            images, labels = data
            classification = model(images)
            clsloss = cls_loss(classification, labels)
            classification = nn.Softmax(dim=1)(classification)
            pred_result = torch.concat((pred_result, torch.argmax(classification, dim=1)))
            pred_prob = torch.concat((pred_prob, classification))
            true_result = torch.concat((true_result, labels))
            total_loss += clsloss


        pred_result = list(map(int, pred_result.cpu()))
        pred_prob = pred_prob.cpu().tolist()
        true_result = list(map(int, true_result.cpu()))

        accuracy = acc(pred_result, true_result, writer=writer, epoch=epoch)
        if writer != None:
            writer.add_scalar("loss", total_loss.cpu() / len(data_loader), epoch)
    result = pd.DataFrame({"pred": pred_result, "true": true_result, "pred prob": pred_prob})
    # result.to_csv(f"./cls/data/output/ResNet3D/{tag}/test_result_{epoch}.csv")
    return result, accuracy

class MyData(Dataset):
    def __init__(self, data_label: pd.DataFrame, label_list: list, device='CPU', modality="PETCT", root_dir=r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'):
        self.data = data_label[data_label['training_label'].isin(label_list)]
        self.data_label = data_label
        self.root_dir = root_dir
        self.device = device
        self.modality = modality

    def __getitem__(self, item):
        # sex_dict = {'F': 0, 'M': 1}
        label_dict = {'NEGATIVE': 0, 'LYMPHOMA': 1, 'MELANOMA': 2, 'LUNG_CANCER': 3}
        patient_id = self.data.iloc[item, 1]
        diagnose = self.data.iloc[item, 0]
        patient_dir = os.path.join(self.root_dir, diagnose, patient_id)
        img_method = "tensor"
        if img_method == "sitk":
            ct_path = os.path.join(patient_dir, 'CT_nii', '1.nii')
            pet_path = os.path.join(patient_dir, 'PET_nii', '1.nii')
            ct_image = normalized(sitk.GetArrayFromImage(sitk.ReadImage(ct_path)), device=self.device)
            pet_image = normalized(sitk.GetArrayFromImage(sitk.ReadImage(pet_path)), device=self.device)
        elif img_method == 'tensor':
            ct_path = os.path.join(patient_dir, 'CT_pt', '1.pt')
            pet_path = os.path.join(patient_dir, 'PET_pt', '1.pt')
            ct_image = normalized(torch.load(ct_path), device=self.device)
            pet_image = normalized(torch.load(pet_path), device=self.device)
        elif img_method == "pickle":
            ct_path = os.path.join(patient_dir, 'ct_pickle', '1.pickle')
            pet_path = os.path.join(patient_dir, 'pet_pickle', '1.pickle')
            ct_f = open(ct_path, 'rb')
            pet_f = open(pet_path, 'rb')
            ct_image = normalized(pickle.load(ct_f), device=self.device)
            pet_image = normalized(pickle.load(pet_f), device=self.device)


        clinical = pd.read_csv(os.path.join(patient_dir, "clinical.csv"))

        label = tensor(label_dict[clinical.loc[0, "label"]]).to(self.device)
        if self.modality == "PETCT":
            image = torch.stack((ct_image, pet_image))
        elif self.modality == "PET":
            image = pet_image.unsqueeze(dim=0)
        elif self.modality == "CT":
            image = ct_image.unsqueeze(dim=0)
        else:
            raise Exception(f"Wrong input: modality={self.modality}, must be one of 'PETCT'/'CT'/'PET'")
        return image, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("running on GPU")
    else:
        device = torch.device('cpu')
        print("running on CPU")

    start = time.time()
    torch.multiprocessing.set_start_method(method='forkserver', force=True)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    persistent_workers = True
    nums_worker = 7
    root_dir = r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'
    # root_dir = r'/data/zhxie/oufu_data_400G/preprocessed'
    # root_dir = r'E:\dataset\preprocessed'

    kFold = 5
    labels = ["NEGATIVE", "LUNG_CANCER", "LYMPHOMA", "MELANOMA"]
    dfs = [pd.DataFrame({"class": [label] * len(os.listdir(os.path.join(root_dir, label))), "name": os.listdir(os.path.join(root_dir, label))}) for label in labels]
    Whole_df = pd.concat(dfs)
    classes = split_train_val_test(clinical=Whole_df, kFold=kFold, test_radio=0.1)
    torch.set_default_dtype(torch.float32)    

    model_names = [
        i+"_"+j for i in ["ResNet3D", "DenseNet3D", "ViT3D"] for j in ["PETCT", "CT", "PET"]
    ]
    for model_modality in model_names:
        if model_modality == "ResNet3D_PETCT":
            continue
        # model_name = "ViT3D_CT"
        model_name = model_modality.split("_")[0]
        modality = model_modality.split("_")[1]
        print(model_name, modality)
        if modality == "PETCT":
            in_chans = 2
        else:
            in_chans = 1

        # for SwinTransformer
        grid_search = {
            'lr': [1e-6, 2e-6],
            'WeightDecay': [2e-4],
            'depth': [4, ],
            'dim': [1024, ],
            'threshold': [0.2, 0.4],
            'epoch': [100],
        }
        
        # for ResNet
        # grid_search = {
        #     'lr': [5e-4,],
        #     'WeightDecay': [0.02, ],
        #     'depth': [2, ],
        #     'PoolSize': [4, ],
        #     'threshold': [0.15, ],
        #     'epoch': [60],
        # }

        batch = 4
        # seg_loss = sigmoid_focal_loss().to(device)
        cls_loss = nn.CrossEntropyLoss().to(device)

        test_data = MyData(classes, [-1], device, root_dir=root_dir, modality=modality)
        test_dataloader = DataLoader(dataset=test_data, batch_size=batch, shuffle=False, drop_last=False, num_workers=nums_worker, persistent_workers=persistent_workers)
        trainLossList, ValLossList = [], []

        train_for_best = True
        if not train_for_best:
            grid_search_path = os.path.join("./cls/Performance", model_modality)

            if os.path.exists(os.path.join(grid_search_path, "grid search.csv")):
                validation_df = pd.read_csv(os.path.join(grid_search_path, "grid search.csv"), index_col=0)
            else:
                validation_df = pd.DataFrame()

            combinations = list(itertools.product(*grid_search.values()))
            for combination in combinations:
                params = dict(zip(tuple(grid_search.keys()), combination))

                acc_list = []
                # for validation in range(kFold):
                validation = 0      # only validation on the first fold

                print(f"start training on valid {validation}")

                param_path = "_".join([str(item) for key_value in params.items() for item in key_value])
                paths = [
                    os.path.join('./cls/log/train', model_modality, param_path, "val"+str(validation)),
                    os.path.join('./cls/log/test', model_modality, param_path, "val"+str(validation)),
                ]

                print(paths)
                for dir in paths:
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                train_writer = SummaryWriter(paths[0])
                test_writer = SummaryWriter(paths[1])

                trainlist = list(range(validation))
                trainlist.extend(list(range(validation+1, kFold)))
                train_data = MyData(classes, trainlist, device, root_dir=root_dir, modality=modality)
                train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False, num_workers=nums_worker, persistent_workers=persistent_workers)
                valid_data = MyData(classes, [validation], device, root_dir=root_dir, modality=modality)
                valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch, shuffle=False, drop_last=False, num_workers=nums_worker, persistent_workers=persistent_workers)

                if model_name == "ResNet3D":
                    model = ResNet18(img_channels=in_chans, depth=params['depth'], avgpool_size=params['PoolSize']).to(device)
                elif model_name == "DenseNet3D":
                    model = DenseNet121_3d(depth=params['depth'], avgpool_size=params['PoolSize'], in_chans=in_chans).to(device)
                elif model_name == "ViT3D":
                    model = ViT(depth=params['depth'], dim=params['dim'], in_chans=in_chans).to(device)
                elif model_name == "SwinTransformer3D":
                    model = SwinTransformer(depth=params['depth'], in_chans=in_chans).to(device)
                else:
                    raise Exception("Wrong model name! Should be one of ResNet3D/DenseNet3D/ViT3D/SwinTransformer3D")

                optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['WeightDecay'])    # L2 loss

                early_stop, model, _, trainLossList, max_acc = training(model, optimizer, cls_loss, train_dataloader, valid_dataloader, verbose=True, writer={"train": train_writer, "test": test_writer}, tag="val"+str(validation), EPOCHS=params['epoch'], threshold=params['threshold'])
                print(f"start validation on valid {validation}")
                # _, accuracy = val(model, valid_dataloader, cls_loss)

                acc_list.append(max_acc)
                print(early_stop, acc_list)

                accuracy = np.mean(np.array(acc_list))

                params['accuracy'] = accuracy
                params['epoch'] = early_stop
                validation_df = validation_df.append(params, ignore_index=True)

                if not os.path.exists(grid_search_path):
                    os.makedirs(grid_search_path)
                validation_df.to_csv(os.path.join(grid_search_path, "grid search.csv"))

        # retrain for best
        try:
            grid_search_path = os.path.join("./cls/Performance", model_modality)
            validation_df = pd.read_csv(os.path.join(grid_search_path, "grid search.csv"), index_col=0)

            print(validation_df)
            params = validation_df.loc[validation_df['accuracy'].idxmax()].to_dict()
            params.pop('accuracy')
        except:
            combination = list(itertools.product(*grid_search.values()))[0]
            params = dict(zip(tuple(grid_search.keys()), combination))

        print(f"best param: {params}")
        param_path = "_".join([str(item) for key_value in params.items() for item in key_value])
        paths = [
            os.path.join('./cls/log/train', model_modality, param_path, "best"),
            os.path.join('./cls/log/test', model_modality, param_path, "best"),
            os.path.join("./cls/Performance", model_modality, param_path, "best"),
            os.path.join("./cls/data/output", model_modality, param_path, "best")
        ]
        for dir in paths:
            if not os.path.exists(dir):
                os.makedirs(dir)
        print(paths)

        train_writer = SummaryWriter(paths[0])
        test_writer = SummaryWriter(paths[1])
        train_data = MyData(classes, list(range(kFold)), device=device, root_dir=root_dir, modality=modality)
        train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False, num_workers=nums_worker, persistent_workers=persistent_workers)

        if model_name == "ResNet3D":
            model = ResNet18(img_channels=in_chans, depth=params['depth'], avgpool_size=params['PoolSize']).to(device)
        elif model_name == "DenseNet3D":
            model = DenseNet121_3d(depth=params['depth'], avgpool_size=params['PoolSize'], in_chans=in_chans).to(device)
        elif model_name == "ViT3D":
            model = ViT(depth=params['depth'], dim=params['dim'], in_chans=in_chans).to(device)
        elif model_name == "SwinTransformer3D":
            model = SwinTransformer(depth=params['depth'], in_chans=in_chans).to(device)
        else:
            raise Exception("Wrong model name! Should be one of ResNet3D/DenseNet3D/ViT3D/SwinTransformer3D")

        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['WeightDecay'])    # L2 loss

        _, model, _, trainLossList, _ = training(model, optimizer, cls_loss, train_dataloader, test_dataloader, verbose=True, writer={"train": train_writer, "test": test_writer}, EPOCHS=params['epoch'], threshold=params['threshold'])
        print("get test result")
        test_result, test_accuracy = val(model, test_dataloader, cls_loss)
        print("get train result")
        train_result, train_accuracy = val(model, train_dataloader, cls_loss)

        torch.save(model, os.path.join('./cls/model', model_modality+'.pt'))
        trainLossList.to_csv(os.path.join(paths[2], "trainLoss.csv"))
        test_result.to_csv(os.path.join(paths[3], "test_result.csv"))
        train_result.to_csv(os.path.join(paths[3], "train_result.csv"))
        print("-"*20, f"\nTest accuracy: {test_accuracy}")
        # # writer.close()
        print(f"total time: {(time.time() - start) / 60: .2f}min")
