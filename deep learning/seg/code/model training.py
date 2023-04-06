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
from models.VNet import VNet

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
    clinical.to_csv("./seg/data/training label.csv")
    return clinical

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = m1 * m2
    dice = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    return dice.sum() / num

def normalized(img, range1=(0, 1), device="cuda"):
    img = torch.as_tensor(img, dtype=torch.float32, device=device)      # int32 to float32
    if img.max() != img.min():
        img = (img - img.min()) / (img.max() - img.min())    # to 0--1
    img = img * (range1[1] - range1[0]) + range1[0]
    return img

def compute_loss(seg_loss, weights: list, outputs: list, mask, device='cuda'):
    total_loss = torch.as_tensor(0., dtype=torch.float32, device=device)
    if len(weights) != len(outputs):
        raise Exception("The len of weights should equal to the len of outputs")
    for weight, output in zip(weights, outputs):
        # print(output.shape, mask.shape)
        total_loss = total_loss + weight * seg_loss(output, mask)
        mask = F.interpolate(mask, scale_factor=0.5)
    return total_loss

def training(model: nn.modules, optimizer, seg_loss, train_loader: DataLoader, test_loader: DataLoader, writer: dict, verbose=False, EPOCHS=50, threshold=0.1, tag="val0"):
    model.train()
    with torch.enable_grad():
        if verbose:
            loss_list = []
        early_stop, step, max_dice = 0, 0, torch.as_tensor(0., device=device)
        for epoch in range(int(EPOCHS)):
            epo_loss_list = []

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            loop = tqdm(train_loader, total=len(train_loader))
            for data in loop:
                step += 1
                images, masks, _ = data
                # image: [从头向下, 从前向后, 从左向右]
                optimizer.zero_grad()
                segmentation = model(images)
                # train_loss = torch.abs(seg_loss(segmentation, masks) - threshold) + threshold   # weightfocal loss
                weights = [2**(-i) for i in range(len(segmentation))]
                weights_one = torch.as_tensor([i/sum(weights) for i in weights], dtype=torch.float32, device='cuda')
                train_loss = compute_loss(seg_loss, weights_one, segmentation, masks)

                train_loss.backward()
                optimizer.step()

                epo_loss_list.append(float(train_loss.cpu()))
                if step % 5 == 0 and len(epo_loss_list) >= 5:
                    writer["train"].add_scalar("loss", sum(epo_loss_list[-5:]) / 5, step)

                loop.set_description(f'Epoch [{epoch + 1}/{EPOCHS}]')

            dice = val(model, test_loader, seg_loss, writer["test"], (epoch+1)*len(loop), tag=tag)

            if verbose:
                loss_list.extend(epo_loss_list)
            
            if max_dice < dice:
                max_dice = dice
                early_stop = epoch
                torch.save(model, os.path.join('./seg/model', f"{model_name}_{tag}_best.pt"))


    if verbose:
        return early_stop, model, train_loss, pd.DataFrame(loss_list), max_dice
    else:
        return early_stop, model, train_loss, max_dice

def val(model, data_loader: DataLoader, seg_loss, writer=None, epoch=50, device="cuda", save=False, tag="val0"):
    model.eval()
    with torch.no_grad():
        total_loss = torch.as_tensor(0., device=device)
        loop = tqdm(data_loader, total=len(data_loader))
        dice = torch.as_tensor(0., device=device)
        for data in loop:
            images, masks, paths = data
            segmentation = model(images)
            weights = [2**(-i) for i in range(len(segmentation))]
            weights_one = [i/sum(weights) for i in weights]
            segloss = compute_loss(seg_loss, weights_one, segmentation, masks)
                 
            sigmoid = nn.Sigmoid()
            segmentation = sigmoid(segmentation[0])
            dice += dice_coeff(segmentation, masks)
            if save:
                segmentation = segmentation.ge(0.5)
                for seg, path in zip(segmentation, paths):
                    seg = torch.squeeze(seg)
                    path = os.path.join(path, 'predict_mask_pt')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(seg, os.path.join(path, '1.pt'))
            total_loss += segloss

        if writer != None:
            writer.add_scalar("loss", total_loss.cpu() / len(data_loader), epoch)
            writer.add_scalar("dice", dice.cpu() / len(data_loader), epoch)
    return dice

class MyData(Dataset):
    def __init__(self, data_label: pd.DataFrame, label_list: list, device='CPU', modality="PETCT", root_dir=r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'):
        self.data = data_label[data_label['training_label'].isin(label_list)]
        self.data_label = data_label
        self.root_dir = root_dir
        self.device = device
        self.modality = modality

    def __getitem__(self, item):
        patient_id = self.data.iloc[item, 1]
        diagnose = self.data.iloc[item, 0]
        patient_dir = os.path.join(self.root_dir, diagnose, patient_id)
        img_method = "tensor"
        if img_method == "sitk":
            ct_path = os.path.join(patient_dir, 'CT_nii', '1.nii')
            pet_path = os.path.join(patient_dir, 'PET_nii', '1.nii')
            seg_path = os.path.join(patient_dir, 'seg_nii', '1.nii')
            ct_image = normalized(sitk.GetArrayFromImage(sitk.ReadImage(ct_path)), device=self.device)
            pet_image = normalized(sitk.GetArrayFromImage(sitk.ReadImage(pet_path)), device=self.device)
            seg = torch.as_tensor(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)), dtype=torch.float32, device=self.device)
        elif img_method == 'tensor':
            ct_path = os.path.join(patient_dir, 'CT_pt', '1.pt')
            pet_path = os.path.join(patient_dir, 'PET_pt', '1.pt')
            seg_path = os.path.join(patient_dir, 'seg_pt', '1.pt')
            ct_image = normalized(torch.load(ct_path), device=self.device)
            pet_image = normalized(torch.load(pet_path), device=self.device)
            seg = torch.as_tensor(torch.load(seg_path), dtype=torch.float32, device=self.device)
        elif img_method == "pickle":
            ct_path = os.path.join(patient_dir, 'ct_pickle', '1.pickle')
            pet_path = os.path.join(patient_dir, 'pet_pickle', '1.pickle')
            seg_path = os.path.join(patient_dir, 'seg_pickle', '1.pickle')
            ct_f = open(ct_path, 'rb')
            pet_f = open(pet_path, 'rb')
            seg_f = open(seg_path, 'rb')
            ct_image = normalized(pickle.load(ct_f), device=self.device)
            pet_image = normalized(pickle.load(pet_f), device=self.device)
            seg = torch.as_tensor(pickle.load(seg_f), dtype=torch.float32, device=self.device)

        seg = torch.where(seg==255., 1., 0.)
        seg = torch.unsqueeze(seg, dim=0)
        if self.modality == "PETCT":
            image = torch.stack((ct_image, pet_image))
        elif self.modality == "PET":
            image = pet_image.unsqueeze(dim=0)
        elif self.modality == "CT":
            image = ct_image.unsqueeze(dim=0)
        else:
            raise Exception(f"Wrong input: modality={self.modality}, must be one of 'PETCT'/'CT'/'PET'")
        return image, seg, patient_dir


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
    nums_worker = 0
    if nums_worker > 0:
        persistent_workers = True
    else:
        persistent_workers = False

    root_dir = r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'
    # root_dir = r'/data/zhxie/oufu_data_400G/preprocessed'
    # root_dir = r'E:\dataset\preprocessed'

    kFold = 5
    labels = ["LUNG_CANCER", "LYMPHOMA", "MELANOMA"]
    dfs = [pd.DataFrame({"class": [label] * len(os.listdir(os.path.join(root_dir, label))), "name": os.listdir(os.path.join(root_dir, label))}) for label in labels]
    Whole_df = pd.concat(dfs)
    classes = split_train_val_test(clinical=Whole_df, kFold=kFold, test_radio=0.1)
    torch.set_default_dtype(torch.float32)

    model_name = "VNet3D_PETCT"

    modality = model_name.split("_")[1]
    if modality == "PETCT":
        in_chans = 2
    else:
        in_chans = 1

    grid_search = {
        'lr': [0.01,],
        'WeightDecay': [3e-5, ],
        'threshold': [0, ],
        'depth': [2, ],
        'epoch': [100],
    }

    batch = 2
    seg_loss = loss.HybridLoss()

    test_data = MyData(classes, [-1], device, root_dir=root_dir, modality=modality)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch, shuffle=False, drop_last=False, num_workers=nums_worker, persistent_workers=persistent_workers)
    trainLossList, ValLossList = [], []

    grid_search_path = os.path.join("./seg/Performance", model_name)

    train_for_best = True
    if not train_for_best:
        if os.path.exists(os.path.join(grid_search_path, "grid search.csv")):
            validation_df = pd.read_csv(os.path.join(grid_search_path, "grid search.csv"), index_col=0)
        else:
            validation_df = pd.DataFrame()
        
        combinations = list(itertools.product(*grid_search.values()))
        for combination in combinations:
            params = dict(zip(tuple(grid_search.keys()), combination))
            print(params)
            dice_list = []
            # for validation in range(kFold):
            validation = 0      # only validation on the first fold

            print(f"start training on valid {validation}")

            param_path = "_".join([str(item) for key_value in params.items() for item in key_value])
            paths = [
                os.path.join('./seg/log/train', model_name, param_path, "val"+str(validation)),
                os.path.join('./seg/log/test', model_name, param_path, "val"+str(validation)),
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

            model = VNet(inChans=in_chans, depth=params['depth']).to(device=device)

            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['WeightDecay'])
            # optimizer = optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['WeightDecay'], momentum=0.99, nesterov=True)

            early_stop, model, _, trainLossList, max_dice = training(model, optimizer, seg_loss, train_dataloader, valid_dataloader, verbose=True, writer={"train": train_writer, "test": test_writer}, tag="val"+str(validation), EPOCHS=params['epoch'], threshold=params['threshold'])
            
            dice_list.append(max_dice.cpu())
            print(dice_list)
            dice = np.mean(np.array(dice_list))

            params['dice'] = dice
            params['epoch'] = early_stop
            validation_df = validation_df.append(params, ignore_index=True)

            if not os.path.exists(grid_search_path):
                os.makedirs(grid_search_path)
            validation_df.to_csv(os.path.join(grid_search_path, "grid search.csv"))

    # retrain for best
    print("Train for best")
    grid_search_path = os.path.join("./seg/Performance", model_name)
    validation_df = pd.read_csv(os.path.join(grid_search_path, "grid search.csv"), index_col=0)

    # print(validation_df)
    # params = validation_df.loc[validation_df['dice'].idxmax()].to_dict()
    # params.pop('dice')
    
    params = {
        'lr': 0.01,
        'WeightDecay': 3e-5,
        'threshold': 0,
        'epoch': 200,
        'depth': 4,
    }

    print(f"best param: {params}")
    param_path = "_".join([str(item) for key_value in params.items() for item in key_value])
    paths = [
        os.path.join('./seg/log/train', model_name, param_path, "best"),
        os.path.join('./seg/log/test', model_name, param_path, "best"),
        os.path.join("./seg/Performance", model_name, param_path, "best"),
        os.path.join("./seg/data/output", model_name, param_path, "best")
    ]

    for dir in paths:
        if not os.path.exists(dir):
            os.makedirs(dir)
    print(paths)
    
    train_writer = SummaryWriter(paths[0])
    test_writer = SummaryWriter(paths[1])
    train_data = MyData(classes, list(range(kFold)), device=device, root_dir=root_dir, modality=modality)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, drop_last=False, num_workers=nums_worker, persistent_workers=persistent_workers)

    model = VNet(inChans=in_chans, depth=params['depth']).to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['WeightDecay'])    # L2 loss

    _, model, _, trainLossList, _ = training(model, optimizer, seg_loss, train_dataloader, test_dataloader, verbose=True, writer={"train": train_writer, "test": test_writer}, EPOCHS=params['epoch'], threshold=params['threshold'])
    print("get test result")
    test_dice = val(model, test_dataloader, seg_loss, save=True)
    print("get train result")
    train_dice = val(model, train_dataloader, seg_loss, save=True)

    torch.save(model, os.path.join('./seg/model', model_name+'.pt'))
    trainLossList.to_csv(os.path.join(paths[2], "trainLoss.csv"))
    print("-"*20, f"\nTest dice: {test_dice}")
    # # writer.close()
    print(f"total time: {(time.time() - start) / 60: .2f}min")